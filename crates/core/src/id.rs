//! Identifiers: [`Id128`], the random id of documents and of objects inside them, and [`Sequence`],
//! a thread-safe counter for ids that only need to be unique within one run.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::collections::ArrayString;
use crate::error::{HexError, PlatformError};
use crate::hex;
use crate::platform::Entropy;

/// A 128-bit random identifier.
///
/// Two ids drawn independently collide with a probability that can be ignored (2^-64 after four
/// billion ids), so documents, layers and blobs made on different machines can be merged without a
/// registry. The text form is exactly 32 lower-case hexadecimal digits, most significant first, so
/// every id has one spelling; [`Id128::NIL`] (all zero) means "no id" and is never drawn.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id128(u128);

impl Id128 {
    /// The id that stands for "none".
    pub const NIL: Self = Self(0);

    /// A new random id.
    ///
    /// Fails when the entropy source fails, or keeps returning only zero bytes (a broken source must
    /// not hand every object the same id).
    pub fn random(entropy: &dyn Entropy) -> Result<Self, PlatformError> {
        for _ in 0..4 {
            let mut bytes = [0_u8; 16];
            entropy.fill(&mut bytes)?;
            let id = Self(u128::from_le_bytes(bytes));
            if !id.is_nil() {
                return Ok(id);
            }
        }
        Err(PlatformError::Other("the entropy source returns only zeros".to_owned()))
    }

    /// The id with the given numeric value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// The id's numeric value.
    pub const fn as_u128(self) -> u128 {
        self.0
    }

    /// The id from its 16 bytes in little-endian order, the form binary files store.
    #[must_use]
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self(u128::from_le_bytes(bytes))
    }

    /// The id's 16 bytes in little-endian order.
    pub const fn to_le_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// `true` for [`Id128::NIL`].
    pub const fn is_nil(self) -> bool {
        self.0 == 0
    }

    /// The text form, on the stack.
    pub fn to_hex(self) -> ArrayString<32> {
        hex::encode_u128(self.0)
    }

    /// Parses the text form: exactly 32 digits `0-9a-f`.
    pub fn parse(text: &str) -> Result<Self, HexError> {
        hex::decode_u128(text).map(Self)
    }
}

impl fmt::Display for Id128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Id128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id128({})", self.to_hex())
    }
}

impl FromStr for Id128 {
    type Err = HexError;

    fn from_str(text: &str) -> Result<Self, HexError> {
        Self::parse(text)
    }
}

/// A counter that hands out increasing numbers from any thread.
///
/// The numbers are unique for the life of the counter (it would take centuries of counting to wrap
/// 64 bits) and mean nothing outside the process unless the owner stores them: a document that
/// persists sequence numbers restores its counter with [`Sequence::advance_to`].
#[derive(Debug)]
pub struct Sequence(AtomicU64);

impl Sequence {
    /// A counter whose first number is 1, leaving 0 free to mean "none".
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU64::new(1))
    }

    /// A counter whose first number is `first`.
    #[must_use]
    pub const fn starting_at(first: u64) -> Self {
        Self(AtomicU64::new(first))
    }

    /// The next number. Every call, on any thread, gets a different one.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    /// The number the next call of [`Sequence::next`] would return, if no other thread calls first.
    pub fn peek(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Makes sure every later number is at least `first` (never moves the counter backwards).
    pub fn advance_to(&self, first: u64) {
        self.0.fetch_max(first, Ordering::Relaxed);
    }
}

impl Default for Sequence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{count_in, is_installed};
    use crate::platform::SeededEntropy;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn ids_round_trip_through_text_and_bytes() {
        let id = Id128::from_u128(0x0123_4567_89ab_cdef_0011_2233_4455_6677);
        assert_eq!(id.to_string(), "0123456789abcdef0011223344556677");
        assert_eq!(format!("{id:?}"), "Id128(0123456789abcdef0011223344556677)");
        assert_eq!(Id128::parse(&id.to_hex()), Ok(id));
        assert_eq!("0123456789abcdef0011223344556677".parse::<Id128>(), Ok(id));
        assert_eq!(id.to_le_bytes()[0], 0x77);
        assert_eq!(Id128::from_le_bytes(id.to_le_bytes()), id);
        assert_eq!(id.as_u128() >> 120, 0x01);
        assert_eq!(Id128::NIL.to_string(), "0".repeat(32));
        assert!(Id128::NIL.is_nil() && Id128::default().is_nil() && !id.is_nil());
        assert!(Id128::from_u128(1) < Id128::from_u128(2));
    }

    // [ROB-01] Ids come from files and from the network: malformed text is an error.
    #[test]
    fn malformed_id_text_is_rejected() {
        assert_eq!(Id128::parse(""), Err(HexError::Length { expected: 32, found: 0 }));
        assert_eq!(Id128::parse(&"a".repeat(31)), Err(HexError::Length { expected: 32, found: 31 }));
        assert_eq!(Id128::parse(&"a".repeat(33)), Err(HexError::Length { expected: 32, found: 33 }));
        assert_eq!(Id128::parse("0123456789ABCDEF0011223344556677"), Err(HexError::Digit { index: 10 }));
        assert_eq!(Id128::parse("0123456789abcdef001122334455667g"), Err(HexError::Digit { index: 31 }));
        assert_eq!(Id128::parse("0123456789abcdef00112233445566é"), Err(HexError::Digit { index: 30 }));
        assert!(" 123456789abcdef0011223344556677".parse::<Id128>().is_err());
    }

    #[test]
    fn random_ids_are_distinct_reproducible_per_seed_and_never_nil() {
        let entropy = SeededEntropy::new(42);
        let mut seen = HashSet::new();
        for _ in 0..10_000 {
            let id = Id128::random(&entropy).expect("id");
            assert!(!id.is_nil());
            assert!(seen.insert(id), "duplicate id {id}");
        }
        let again = SeededEntropy::new(42);
        assert!(seen.contains(&Id128::random(&again).expect("id")));
        // All bits are used.
        let (mut ones, mut zeros) = (0_u128, 0_u128);
        for id in &seen {
            ones |= id.as_u128();
            zeros |= !id.as_u128();
        }
        assert_eq!((ones, zeros), (u128::MAX, u128::MAX));
    }

    #[test]
    fn random_id_reports_a_failing_or_stuck_entropy_source() {
        struct Failing;
        impl Entropy for Failing {
            fn fill(&self, _out: &mut [u8]) -> Result<(), PlatformError> {
                Err(PlatformError::Unsupported)
            }
        }
        struct Zeros;
        impl Entropy for Zeros {
            fn fill(&self, out: &mut [u8]) -> Result<(), PlatformError> {
                out.fill(0);
                Ok(())
            }
        }
        assert_eq!(Id128::random(&Failing), Err(PlatformError::Unsupported));
        assert!(matches!(Id128::random(&Zeros), Err(PlatformError::Other(message)) if message.contains("zeros")));
    }

    #[test]
    fn formatting_an_id_does_not_allocate() {
        use std::fmt::Write as _;
        let id = Id128::from_u128(u128::MAX / 3);
        let mut text = ArrayString::<48>::new();
        let stats = count_in(|| {
            let _ = write!(text, "{id} {id:?}");
            std::hint::black_box(id.to_hex());
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert!(text.starts_with("55555555"));
    }

    #[test]
    fn sequence_counts_up_and_never_goes_back() {
        static GLOBAL: Sequence = Sequence::new();
        let seq = Sequence::new();
        assert_eq!((seq.peek(), seq.next(), seq.next(), seq.peek()), (1, 1, 2, 3));
        seq.advance_to(100);
        assert_eq!(seq.next(), 100);
        seq.advance_to(50);
        assert_eq!(seq.next(), 101, "advance_to never moves backwards");
        assert_eq!(Sequence::starting_at(7).next(), 7);
        assert_eq!(Sequence::default().next(), 1);
        assert!(GLOBAL.next() >= 1);
    }

    #[test]
    fn sequence_numbers_are_unique_across_threads() {
        let seq = Arc::new(Sequence::new());
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let seq = Arc::clone(&seq);
                std::thread::spawn(move || (0..5_000).map(|_| seq.next()).collect::<Vec<u64>>())
            })
            .collect();
        let mut all: Vec<u64> = workers.into_iter().flat_map(|w| w.join().expect("join")).collect();
        all.sort_unstable();
        assert_eq!(all, (1..=40_000).collect::<Vec<u64>>());
        assert_eq!(seq.peek(), 40_001);
    }
}
