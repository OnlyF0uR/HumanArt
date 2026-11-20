//! In-house hashing `[DEP-05]`: checksums, the 128-bit content hash and a fast map hasher.
//!
//! - [`crc32`] (IEEE 802.3, the polynomial of PNG, gzip and zip) and [`adler32`] (zlib) are the
//!   checksums the codecs in `humanart-io` need. Both follow zlib's calling convention: `x_update(previous,
//!   more_bytes)` continues a checksum, and the checksum of nothing is 0 for CRC-32 and 1 for Adler-32.
//! - [`Hash128`] is MurmurHash3 x64_128 (Austin Appleby, public domain), the content address of the
//!   blob store `[IO-30]`. It is fast (about one multiply per 8 bytes), has no known practical
//!   collisions on non-adversarial data, and is verified here against the SMHasher verification value
//!   and by a collision test over more than a million short inputs. It is **not** cryptographic: the
//!   store additionally verifies sizes, and nothing security-relevant may rely on it.
//! - [`Hash64`] with [`BuildFast`] is the hasher for in-memory hash maps ([`FastMap`], [`FastSet`]):
//!   a folded-multiply mixer, several times faster than the standard SipHash for the small integer
//!   and short string keys this workspace uses. It is deterministic (no per-process seed), so
//!   anything derived from iteration order is reproducible, and for the same reason it does not
//!   resist hash flooding: a map keyed by text straight from the network keeps the standard hasher.

use crate::collections::ArrayString;
use crate::error::HexError;
use crate::hex;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasher, Hasher};

/// The reflected IEEE 802.3 polynomial.
const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

/// Builds the eight slice-by-8 tables: `tables[0]` is the classic byte table, `tables[k][i]` is the
/// CRC of byte `i` followed by `k` zero bytes.
const fn make_crc32_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0_u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ CRC32_POLYNOMIAL } else { crc >> 1 };
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut crc = tables[0][i];
        let mut k = 1;
        while k < 8 {
            crc = tables[0][(crc & 0xFF) as usize] ^ (crc >> 8);
            tables[k][i] = crc;
            k += 1;
        }
        i += 1;
    }
    tables
}

/// 8 KiB of tables, computed at compile time.
static CRC32_TABLES: [[u32; 256]; 8] = make_crc32_tables();

/// The CRC-32 (IEEE) of `bytes`, as zlib's `crc32(0, bytes)`.
///
/// ```
/// assert_eq!(humanart_core::hash::crc32(b"123456789"), 0xCBF4_3926);
/// ```
#[inline]
pub fn crc32(bytes: &[u8]) -> u32 {
    crc32_update(0, bytes)
}

/// Continues a CRC-32: `crc32_update(crc32(a), b) == crc32(a ++ b)`. Start from 0.
pub fn crc32_update(crc: u32, bytes: &[u8]) -> u32 {
    let t = &CRC32_TABLES;
    let mut crc = !crc;
    let (blocks, rest) = bytes.as_chunks::<8>();
    for block in blocks {
        let low = u32::from_le_bytes([block[0], block[1], block[2], block[3]]) ^ crc;
        let high = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        crc = t[7][(low & 0xFF) as usize]
            ^ t[6][((low >> 8) & 0xFF) as usize]
            ^ t[5][((low >> 16) & 0xFF) as usize]
            ^ t[4][(low >> 24) as usize]
            ^ t[3][(high & 0xFF) as usize]
            ^ t[2][((high >> 8) & 0xFF) as usize]
            ^ t[1][((high >> 16) & 0xFF) as usize]
            ^ t[0][(high >> 24) as usize];
    }
    for &byte in rest {
        crc = t[0][((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// The largest prime below 2^16, the modulus of Adler-32.
const ADLER_MODULUS: u32 = 65_521;
/// The most bytes that can be summed before the 32-bit accumulators must be reduced (zlib's NMAX).
const ADLER_BLOCK: usize = 5552;

/// The Adler-32 checksum of `bytes`, as zlib's `adler32(1, bytes)`.
///
/// ```
/// assert_eq!(humanart_core::hash::adler32(b"Wikipedia"), 0x11E6_0398);
/// ```
#[inline]
pub fn adler32(bytes: &[u8]) -> u32 {
    adler32_update(1, bytes)
}

/// Continues an Adler-32: `adler32_update(adler32(a), b) == adler32(a ++ b)`. Start from 1.
pub fn adler32_update(adler: u32, bytes: &[u8]) -> u32 {
    // Reduce first, so a state that did not come from this function cannot overflow the sums.
    let mut a = (adler & 0xFFFF) % ADLER_MODULUS;
    let mut b = (adler >> 16) % ADLER_MODULUS;
    for block in bytes.chunks(ADLER_BLOCK) {
        for &byte in block {
            a += u32::from(byte);
            b += a;
        }
        a %= ADLER_MODULUS;
        b %= ADLER_MODULUS;
    }
    (b << 16) | a
}

/// A 128-bit content hash (MurmurHash3 x64_128, seed 0 unless stated otherwise).
///
/// The text form is 32 lower-case hexadecimal digits: the first 64-bit half, then the second, each
/// most significant digit first (`hash128(b"hello")` is `cbd8a7b341bd9b025b1e906a48ae1d19`). The
/// ordering (`Ord`) is the ordering of that text.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hash128 {
    h1: u64,
    h2: u64,
}

impl Hash128 {
    /// The all-zero value, which is also the hash of the empty input. Used as "no content".
    pub const ZERO: Self = Self { h1: 0, h2: 0 };

    /// A hash from its two halves, as produced by the reference implementation (`h1`, `h2`).
    #[inline]
    pub const fn from_parts(h1: u64, h2: u64) -> Self {
        Self { h1, h2 }
    }

    /// The first half.
    #[inline]
    pub const fn h1(self) -> u64 {
        self.h1
    }

    /// The second half.
    #[inline]
    pub const fn h2(self) -> u64 {
        self.h2
    }

    /// The value as one integer: `h1` in the high 64 bits.
    #[inline]
    pub const fn to_u128(self) -> u128 {
        ((self.h1 as u128) << 64) | self.h2 as u128
    }

    /// The inverse of [`to_u128`](Self::to_u128).
    #[inline]
    pub const fn from_u128(value: u128) -> Self {
        Self { h1: (value >> 64) as u64, h2: value as u64 }
    }

    /// The 16 bytes whose hexadecimal form is the text form (big-endian `h1`, then big-endian `h2`).
    /// This is the layout binary files use.
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.to_u128().to_be_bytes()
    }

    /// The inverse of [`to_bytes`](Self::to_bytes).
    #[inline]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self::from_u128(u128::from_be_bytes(bytes))
    }

    /// The digest in the byte order of the reference implementation (little-endian `h1`, then
    /// little-endian `h2`), for comparing against other MurmurHash3 implementations.
    pub fn to_reference_digest(self) -> [u8; 16] {
        let mut out = [0_u8; 16];
        out[..8].copy_from_slice(&self.h1.to_le_bytes());
        out[8..].copy_from_slice(&self.h2.to_le_bytes());
        out
    }

    /// `true` for [`Hash128::ZERO`].
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.h1 == 0 && self.h2 == 0
    }

    /// The text form, without allocating.
    pub fn to_hex(self) -> ArrayString<32> {
        hex::encode(&self.to_bytes())
    }

    /// Parses the text form: exactly 32 lower-case hexadecimal digits.
    pub fn parse(text: &str) -> Result<Self, HexError> {
        hex::decode_array::<16>(text).map(Self::from_bytes)
    }

    /// A well-mixed 64-bit value for hash maps keyed by content hashes.
    #[inline]
    pub const fn fold64(self) -> u64 {
        self.h1 ^ self.h2.rotate_left(32)
    }
}

impl fmt::Display for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_hex().as_str())
    }
}

impl fmt::Debug for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash128({})", self.to_hex())
    }
}

impl std::str::FromStr for Hash128 {
    type Err = HexError;

    fn from_str(text: &str) -> Result<Self, HexError> {
        Self::parse(text)
    }
}

const MURMUR_C1: u64 = 0x87C3_7B91_1142_53D5;
const MURMUR_C2: u64 = 0x4CF5_AD43_2745_937F;

/// The 64-bit finaliser of MurmurHash3: a bijection that avalanches every input bit.
#[inline]
const fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    k ^= k >> 33;
    k = k.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    k ^= k >> 33;
    k
}

#[inline]
const fn mix_k1(k1: u64) -> u64 {
    k1.wrapping_mul(MURMUR_C1).rotate_left(31).wrapping_mul(MURMUR_C2)
}

#[inline]
const fn mix_k2(k2: u64) -> u64 {
    k2.wrapping_mul(MURMUR_C2).rotate_left(33).wrapping_mul(MURMUR_C1)
}

/// Up to eight bytes as a little-endian integer, missing high bytes being zero.
#[inline]
fn le_partial(bytes: &[u8]) -> u64 {
    let mut word = [0_u8; 8];
    let take = bytes.len().min(8);
    word[..take].copy_from_slice(&bytes[..take]);
    u64::from_le_bytes(word)
}

/// A streaming MurmurHash3 x64_128: feeding the input in any number of pieces gives the same hash as
/// [`hash128`] over the concatenation.
#[derive(Clone, Debug)]
pub struct Hasher128 {
    h1: u64,
    h2: u64,
    /// The bytes of the current, incomplete 16-byte block.
    pending: [u8; 16],
    pending_len: usize,
    /// Total bytes fed so far.
    total: u64,
}

impl Hasher128 {
    /// A hasher with seed 0, the one content addressing uses.
    #[inline]
    pub const fn new() -> Self {
        Self::with_seed(0)
    }

    /// A hasher with another seed (the reference implementation takes a 32-bit seed), for independent
    /// hash families.
    #[inline]
    pub const fn with_seed(seed: u32) -> Self {
        Self { h1: seed as u64, h2: seed as u64, pending: [0; 16], pending_len: 0, total: 0 }
    }

    #[inline]
    fn block(&mut self, k1: u64, k2: u64) {
        self.h1 ^= mix_k1(k1);
        self.h1 = self.h1.rotate_left(27).wrapping_add(self.h2).wrapping_mul(5).wrapping_add(0x52DC_E729);
        self.h2 ^= mix_k2(k2);
        self.h2 = self.h2.rotate_left(31).wrapping_add(self.h1).wrapping_mul(5).wrapping_add(0x3849_5AB5);
    }

    /// Feeds more input.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.total = self.total.wrapping_add(bytes.len() as u64);
        if self.pending_len > 0 {
            let take = bytes.len().min(16 - self.pending_len);
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&bytes[..take]);
            self.pending_len += take;
            bytes = &bytes[take..];
            if self.pending_len < 16 {
                return;
            }
            let block = self.pending;
            self.block(le_partial(&block[..8]), le_partial(&block[8..]));
            self.pending_len = 0;
        }
        let (blocks, rest) = bytes.as_chunks::<16>();
        for block in blocks {
            self.block(le_partial(&block[..8]), le_partial(&block[8..]));
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    /// The hash of everything fed so far. The hasher can keep receiving input afterwards.
    pub fn finish(&self) -> Hash128 {
        let (mut h1, mut h2) = (self.h1, self.h2);
        let tail = &self.pending[..self.pending_len];
        if tail.len() > 8 {
            h2 ^= mix_k2(le_partial(&tail[8..]));
        }
        if !tail.is_empty() {
            h1 ^= mix_k1(le_partial(tail));
        }
        h1 ^= self.total;
        h2 ^= self.total;
        h1 = h1.wrapping_add(h2);
        h2 = h2.wrapping_add(h1);
        h1 = fmix64(h1);
        h2 = fmix64(h2);
        h1 = h1.wrapping_add(h2);
        h2 = h2.wrapping_add(h1);
        Hash128 { h1, h2 }
    }
}

impl Default for Hasher128 {
    fn default() -> Self {
        Self::new()
    }
}

/// The content hash of `bytes` (MurmurHash3 x64_128, seed 0) `[IO-30]`.
///
/// ```
/// assert_eq!(humanart_core::hash::hash128(b"hello").to_string(), "cbd8a7b341bd9b025b1e906a48ae1d19");
/// ```
pub fn hash128(bytes: &[u8]) -> Hash128 {
    hash128_seeded(bytes, 0)
}

/// MurmurHash3 x64_128 of `bytes` with the given seed.
pub fn hash128_seeded(bytes: &[u8], seed: u32) -> Hash128 {
    let mut hasher = Hasher128::with_seed(seed);
    hasher.update(bytes);
    hasher.finish()
}

/// The multiplier of the mixing steps: 2^64 divided by the golden ratio, odd.
const FOLD_STEP: u64 = 0x9E37_79B9_7F4A_7C15;
/// The multiplier of the final mix, an unrelated odd constant with balanced bits.
const FOLD_FINISH: u64 = 0xD6E8_FEB8_6659_FD93;
/// The initial state: fractional digits of pi, so that hashing zero is not a fixed point.
const FOLD_INIT: u64 = 0x243F_6A88_85A3_08D3;

/// The folded multiply: the 128-bit product of the operands with its halves xored together. Every
/// input bit reaches both the top and the bottom bits of the result.
#[inline]
const fn fold(a: u64, b: u64) -> u64 {
    let product = (a as u128).wrapping_mul(b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

/// The fast, deterministic 64-bit hasher for in-memory maps (see the module comment for when not to
/// use it). Implements [`std::hash::Hasher`].
#[derive(Clone, Copy, Debug)]
pub struct Hash64 {
    state: u64,
}

impl Hash64 {
    /// A hasher in its initial state.
    #[inline]
    pub const fn new() -> Self {
        Self { state: FOLD_INIT }
    }

    #[inline]
    fn word(&mut self, word: u64) {
        self.state = fold(self.state ^ word, FOLD_STEP);
    }
}

impl Default for Hash64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for Hash64 {
    #[inline]
    fn finish(&self) -> u64 {
        fold(self.state, FOLD_FINISH)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // The length goes in first, so inputs that differ only in trailing zero bytes differ.
        self.word((bytes.len() as u64).wrapping_mul(FOLD_FINISH));
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.word(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            self.word(le_partial(rest));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.word(u64::from(value));
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.word(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.word(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.word(value);
    }

    #[inline]
    fn write_u128(&mut self, value: u128) {
        self.word(value as u64);
        self.word((value >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.word(value as u64);
    }
}

/// The [`BuildHasher`] that makes [`Hash64`] hashers: `HashMap::with_hasher(BuildFast)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildFast;

impl BuildHasher for BuildFast {
    type Hasher = Hash64;

    #[inline]
    fn build_hasher(&self) -> Hash64 {
        Hash64::new()
    }
}

/// A `HashMap` with the fast hasher. Create one with `FastMap::default()`.
pub type FastMap<K, V> = HashMap<K, V, BuildFast>;

/// A `HashSet` with the fast hasher. Create one with `FastSet::default()`.
pub type FastSet<T> = HashSet<T, BuildFast>;

/// The fast 64-bit hash of `bytes` (not stable across versions of this crate: never store it).
#[inline]
pub fn hash64(bytes: &[u8]) -> u64 {
    let mut hasher = Hash64::new();
    hasher.write(bytes);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The textbook bit-at-a-time CRC-32, independent of the tables.
    fn crc32_bitwise(bytes: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }

    /// Adler-32 with a reduction after every byte.
    fn adler32_simple(bytes: &[u8]) -> u32 {
        let (mut a, mut b) = (1_u32, 0_u32);
        for &byte in bytes {
            a = (a + u32::from(byte)) % 65_521;
            b = (b + a) % 65_521;
        }
        (b << 16) | a
    }

    fn random_bytes(rng: &mut Rng, len: usize) -> Vec<u8> {
        let mut bytes = vec![0_u8; len];
        rng.fill(&mut bytes);
        bytes
    }

    // [DEP-05] Values from zlib 1.3 (`zlib.crc32` of CPython), which defines the convention.
    #[test]
    fn crc32_matches_zlib_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
        let ramp: Vec<u8> = (0..=255_u8).cycle().take(256 * 40).collect();
        assert_eq!(crc32(&ramp), 0xBBCE_3B9D);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn crc32_slice_by_8_matches_the_bitwise_definition_at_every_length_and_split() {
        let mut rng = Rng::new(0xC2C);
        let data = random_bytes(&mut rng, 300);
        for len in 0..data.len() {
            assert_eq!(crc32(&data[..len]), crc32_bitwise(&data[..len]), "length {len}");
        }
        for _ in 0..300 {
            let len = rng.index(5000);
            let bytes = random_bytes(&mut rng, len);
            let expect = crc32_bitwise(&bytes);
            assert_eq!(crc32(&bytes), expect);
            let (a, b) = (rng.index(bytes.len() + 1), rng.index(bytes.len() + 1));
            let (first, second) = (a.min(b), a.max(b));
            let mut crc = crc32_update(0, &bytes[..first]);
            crc = crc32_update(crc, &bytes[first..second]);
            crc = crc32_update(crc, &bytes[second..]);
            assert_eq!(crc, expect);
        }
    }

    // [DEP-05] Values from zlib 1.3.
    #[test]
    fn adler32_matches_zlib_vectors() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(&vec![b'a'; 100_000]), 0x7966_0B4D);
        assert_eq!(adler32(b"a"), 0x0062_0062);
    }

    #[test]
    fn adler32_never_overflows_and_streams() {
        // The worst case for the deferred reduction: long runs of 0xFF.
        let worst = vec![0xFF_u8; 3 * ADLER_BLOCK + 17];
        assert_eq!(adler32(&worst), adler32_simple(&worst));
        // A state that no valid stream produces must still not overflow (wrapping would panic in
        // this build, which has overflow checks).
        let _ = adler32_update(0xFFFF_FFFF, &worst);
        let mut rng = Rng::new(0xAD1E);
        for _ in 0..200 {
            let len = rng.index(20_000);
            let bytes = random_bytes(&mut rng, len);
            let expect = adler32_simple(&bytes);
            assert_eq!(adler32(&bytes), expect);
            let cut = rng.index(bytes.len() + 1);
            assert_eq!(adler32_update(adler32(&bytes[..cut]), &bytes[cut..]), expect);
        }
    }

    // [IO-30] The SMHasher verification procedure: hash the keys {}, {0}, {0,1}, ... of length 0..255
    // with seed 256 - length, hash the concatenated digests with seed 0, and read the first four
    // digest bytes as a little-endian integer. SMHasher lists 0x6384BA69 for MurmurHash3_x64_128.
    #[test]
    fn io_30_hash128_matches_the_smhasher_verification_value() {
        let key: Vec<u8> = (0..=255_u8).collect();
        let mut digests = Vec::with_capacity(256 * 16);
        for len in 0..256_usize {
            let hash = hash128_seeded(&key[..len], (256 - len) as u32);
            digests.extend_from_slice(&hash.to_reference_digest());
        }
        let last = hash128(&digests).to_reference_digest();
        let verification = u32::from_le_bytes([last[0], last[1], last[2], last[3]]);
        assert_eq!(verification, 0x6384_BA69);
    }

    // [IO-30] Published vectors (seed 0), as printed by the reference implementation's users.
    #[test]
    fn io_30_hash128_known_vectors() {
        assert_eq!(hash128(b""), Hash128::ZERO);
        assert_eq!(hash128(b"hello").to_string(), "cbd8a7b341bd9b025b1e906a48ae1d19");
        assert_eq!(
            hash128(b"The quick brown fox jumps over the lazy dog").to_string(),
            "e34bbc7bbc071b6c7a433ca9c49a9347"
        );
        assert_eq!(hash128_seeded(b"", 1).to_string(), "4610abe56eff5cb551622daa78f83583");
        assert_eq!(hash128(b"hello").h1(), 0xCBD8_A7B3_41BD_9B02);
        assert_eq!(hash128(b"hello").h2(), 0x5B1E_906A_48AE_1D19);
    }

    #[test]
    fn hash128_streaming_equals_one_shot_for_every_split() {
        let mut rng = Rng::new(0x57);
        let data = random_bytes(&mut rng, 100);
        // Every prefix length, split at every position: covers all block and tail alignments.
        for len in 0..data.len() {
            let expect = hash128(&data[..len]);
            for cut in 0..=len {
                let mut hasher = Hasher128::new();
                hasher.update(&data[..cut]);
                hasher.update(&data[cut..len]);
                assert_eq!(hasher.finish(), expect, "length {len}, cut {cut}");
            }
        }
        for _ in 0..200 {
            let len = rng.index(3000);
            let bytes = random_bytes(&mut rng, len);
            let mut hasher = Hasher128::default();
            let mut rest = bytes.as_slice();
            while !rest.is_empty() {
                let take = 1 + rng.index(rest.len().min(70));
                hasher.update(&rest[..take]);
                hasher.update(&[]);
                rest = &rest[take..];
            }
            assert_eq!(hasher.finish(), hash128(&bytes));
            // `finish` does not consume: more input continues the same stream.
            let mut longer = bytes.clone();
            longer.extend_from_slice(b"tail");
            hasher.update(b"tail");
            assert_eq!(hasher.finish(), hash128(&longer));
        }
    }

    // [IO-30] Collision test: more than a million distinct short inputs of several shapes. A 128-bit
    // hash of this quality must show no collision at all, none in either 64-bit half, and a number of
    // 32-bit collisions close to the birthday expectation n^2 / 2^33.
    #[test]
    fn io_30_hash128_has_no_collisions_over_a_million_short_inputs() {
        let mut hashes: Vec<Hash128> = Vec::with_capacity(1_300_000);
        // Counters as little-endian integers (differ in few bits). From 1: eight zero bytes are
        // already among the runs below.
        for i in 1..=400_000_u64 {
            hashes.push(hash128(&i.to_le_bytes()));
        }
        // Counters as decimal text (blob names, parameter ids).
        let mut text = ArrayString::<24>::new();
        for i in 0..400_000_u32 {
            use std::fmt::Write as _;
            text.clear();
            let _ = write!(text, "param.{i}");
            hashes.push(hash128(text.as_bytes()));
        }
        // Sparse inputs: one or two bits set in 32 zero bytes, and runs of one byte value.
        for a in 0..256_usize {
            for b in a..256 {
                let mut bytes = [0_u8; 32];
                bytes[a / 8] |= 1 << (a % 8);
                bytes[b / 8] |= 1 << (b % 8);
                hashes.push(hash128(&bytes));
            }
        }
        for len in 1..=1024_usize {
            for value in [0_u8, 1, 0x80, 0xFF] {
                hashes.push(hash128(&vec![value; len]));
            }
        }
        // Random lengths 17..=48 with random content (distinct with overwhelming probability; the
        // counter prefix makes it certain).
        let mut rng = Rng::new(0x10_30);
        for i in 0..400_000_u32 {
            let mut bytes = [0_u8; 48];
            let len = 17 + rng.index(32);
            rng.fill(&mut bytes[..len]);
            bytes[..4].copy_from_slice(&i.to_le_bytes());
            bytes[4] = 0xA5;
            hashes.push(hash128(&bytes[..len]));
        }
        let count = hashes.len();
        assert!(count >= 1_000_000, "{count}");

        let distinct = |mut keys: Vec<u64>| {
            keys.sort_unstable();
            keys.dedup();
            keys.len()
        };
        let mut full = hashes.clone();
        full.sort_unstable();
        full.dedup();
        assert_eq!(full.len(), count, "128-bit collision");
        assert_eq!(distinct(hashes.iter().map(|h| h.h1()).collect()), count, "collision in h1");
        assert_eq!(distinct(hashes.iter().map(|h| h.h2()).collect()), count, "collision in h2");
        assert_eq!(distinct(hashes.iter().map(|h| h.fold64()).collect()), count, "collision in fold64");

        let expected = (count as f64).powi(2) / 2.0_f64.powi(33);
        for (name, shift) in [("low", 0), ("high", 32)] {
            let low = distinct(hashes.iter().map(|h| (h.h1() >> shift) & 0xFFFF_FFFF).collect());
            let collisions = (count - low) as f64;
            assert!(
                collisions > expected * 0.6 && collisions < expected * 1.4,
                "{name} 32 bits of h1: {collisions} collisions, expected about {expected:.0}"
            );
        }
    }

    // [IO-30] Avalanche: flipping any single input bit flips about half of the 128 output bits.
    #[test]
    fn io_30_hash128_avalanches() {
        let mut rng = Rng::new(0xA7A);
        let (mut flipped, mut trials) = (0_u64, 0_u64);
        let mut worst = (128_u32, 0_u32);
        for _ in 0..400 {
            let len = 1 + rng.index(40);
            let mut bytes = random_bytes(&mut rng, len);
            let base = hash128(&bytes).to_u128();
            for bit in 0..bytes.len() * 8 {
                bytes[bit / 8] ^= 1 << (bit % 8);
                let changed = (hash128(&bytes).to_u128() ^ base).count_ones();
                bytes[bit / 8] ^= 1 << (bit % 8);
                flipped += u64::from(changed);
                trials += 1;
                worst = (worst.0.min(changed), worst.1.max(changed));
            }
        }
        let mean = flipped as f64 / trials as f64;
        assert!((mean - 64.0).abs() < 0.5, "mean flipped bits {mean}");
        assert!(worst.0 >= 30 && worst.1 <= 98, "{worst:?}");
    }

    #[test]
    fn hash128_text_and_binary_forms_round_trip() {
        let hash = hash128(b"hello");
        assert_eq!(format!("{hash:?}"), "Hash128(cbd8a7b341bd9b025b1e906a48ae1d19)");
        assert_eq!(hash.to_hex().as_str(), "cbd8a7b341bd9b025b1e906a48ae1d19");
        assert_eq!(Hash128::parse("cbd8a7b341bd9b025b1e906a48ae1d19"), Ok(hash));
        assert_eq!("cbd8a7b341bd9b025b1e906a48ae1d19".parse::<Hash128>(), Ok(hash));
        assert_eq!(Hash128::from_bytes(hash.to_bytes()), hash);
        assert_eq!(Hash128::from_u128(hash.to_u128()), hash);
        assert_eq!(Hash128::from_parts(hash.h1(), hash.h2()), hash);
        assert_eq!(hash.to_bytes()[0], 0xCB);
        assert_eq!(hash.to_reference_digest()[0], 0x02);
        assert!(Hash128::ZERO.is_zero() && !hash.is_zero());
        assert_eq!(Hash128::default(), Hash128::ZERO);
        // Ordering is the ordering of the text.
        let other = hash128(b"world");
        assert_eq!(hash.cmp(&other), hash.to_string().cmp(&other.to_string()));

        assert_eq!(Hash128::parse(""), Err(HexError::Length { expected: 32, found: 0 }));
        assert_eq!(
            Hash128::parse("CBD8A7B341BD9B025B1E906A48AE1D19"),
            Err(HexError::Digit { index: 0 }),
            "upper case is a second spelling and is refused"
        );
        assert_eq!(
            Hash128::parse("cbd8a7b341bd9b025b1e906a48ae1d1"),
            Err(HexError::Length { expected: 32, found: 31 })
        );
        assert_eq!(Hash128::parse("cbd8a7b341bd9b025b1e906a48ae1d1z"), Err(HexError::Digit { index: 31 }));
    }

    // [ROB-01]
    #[test]
    fn hash128_parse_survives_mutated_input() {
        let corpus = vec![b"cbd8a7b341bd9b025b1e906a48ae1d19".to_vec(), b"00000000000000000000000000000000".to_vec()];
        crate::fuzz::run("hash128-parse", &corpus, 3000, 0x1030, std::time::Duration::from_secs(60), |bytes| {
            if let Ok(text) = std::str::from_utf8(bytes)
                && let Ok(hash) = Hash128::parse(text)
            {
                assert_eq!(hash.to_string(), text);
            }
        });
    }

    #[test]
    fn hash64_spreads_sequential_integers_over_both_ends_of_the_word() {
        // hashbrown takes the bucket from the low bits and its control byte from the top 7 bits.
        let build = BuildFast;
        let count = 1_000_000_u64;
        let mut low = vec![0_u32; 1024];
        let mut top = vec![0_u32; 128];
        let mut all = Vec::with_capacity(count as usize);
        for key in 0..count {
            let hash = build.hash_one(key);
            low[(hash & 1023) as usize] += 1;
            top[(hash >> 57) as usize] += 1;
            all.push(hash);
        }
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len() as u64, count, "64-bit collision among sequential integers");
        let check = |buckets: &[u32], what: &str| {
            let mean = count as f64 / buckets.len() as f64;
            // Chi-square against the uniform distribution; for k buckets it is about k +- sqrt(2k).
            let chi: f64 = buckets.iter().map(|&n| (f64::from(n) - mean).powi(2) / mean).sum();
            let k = buckets.len() as f64;
            assert!(chi < k + 6.0 * (2.0 * k).sqrt(), "{what}: chi-square {chi} for {k} buckets");
        };
        check(&low, "low 10 bits");
        check(&top, "top 7 bits");
    }

    #[test]
    fn hash64_distinguishes_lengths_types_and_pieces() {
        assert_ne!(hash64(b""), hash64(&[0]));
        assert_ne!(hash64(&[1]), hash64(&[1, 0]));
        assert_ne!(hash64(&[0; 8]), hash64(&[0; 9]));
        assert_ne!(hash64(&[0; 8]), hash64(&[0; 16]));
        assert_eq!(hash64(b"same"), hash64(b"same"));
        let build = BuildFast;
        assert_ne!(build.hash_one(0_u64), build.hash_one(1_u64));
        assert_ne!(build.hash_one(("ab", "c")), build.hash_one(("a", "bc")));
        assert_ne!(build.hash_one(1_u128), build.hash_one(1_u128 << 64));
        assert_ne!(build.hash_one((1_u8, 2_u16, 3_u32, 4_usize)), build.hash_one((1_u8, 2_u16, 3_u32, 5_usize)));
        // Short strings: no collisions and no clustering among a dictionary-sized set.
        let mut seen = FastSet::default();
        let mut hashes = Vec::new();
        for i in 0..200_000_u32 {
            let name = format!("joint_{i}");
            hashes.push(hash64(name.as_bytes()));
            assert!(seen.insert(name));
        }
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), 200_000);
        assert_eq!(seen.len(), 200_000);
    }

    #[test]
    fn fast_map_behaves_like_a_map() {
        let mut map: FastMap<u32, &str> = FastMap::default();
        map.insert(1, "one");
        map.insert(2, "two");
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.remove(&2), Some("two"));
        assert_eq!(map.len(), 1);
        let mut by_hash: FastMap<Hash128, usize> = HashMap::with_hasher(BuildFast);
        by_hash.insert(hash128(b"a"), 1);
        assert_eq!(by_hash[&hash128(b"a")], 1);
        assert_eq!(Hash64::default().finish(), Hash64::new().finish());
    }
}
