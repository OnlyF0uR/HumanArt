//! JSON numbers: the [`Number`] value, exact decimal-to-double conversion and shortest output.
//!
//! # Reading
//!
//! A literal without a fraction or an exponent that fits `i64` or `u64` is kept as that integer, so
//! ids, counts and offsets never pass through a float. `-0` is the one exception: it becomes the float
//! `-0.0`, because an integer cannot carry the sign. Everything else becomes an `f64` that is the
//! **correctly rounded** (round-half-to-even) value of the decimal text, however many digits it has:
//!
//! 1. up to 19 significant digits with a small exponent are converted with one exact IEEE operation
//!    (Clinger's fast path) or with exact 128-bit integer arithmetic;
//! 2. anything else goes through a fixed-size big integer: `digits × 5^e` is formed exactly (or
//!    `digits × 2^k` is divided exactly by `5^-e`, the remainder kept as a sticky bit) and the top 64
//!    bits are rounded once.
//!
//! Only the first [`MAX_SIGNIFICANT_DIGITS`] significant digits are kept; later digits can only matter
//! as "something non-zero follows", which is exactly what the sticky bit records (no rounding boundary
//! of an `f64` has more than 767 significant decimal digits). A value that rounds to infinity is an
//! error (`1e400` is rejected: the document model has no infinity and the writer never emits one); a
//! value too small for a subnormal rounds to zero and keeps its sign.
//!
//! # Writing
//!
//! Integers are written in full. A float is written with the shortest digit string that reads back
//! as exactly the same `f64`: the digits come from the standard library's `{:e}` formatting, which
//! is documented to be shortest round-trip, and are laid out here as a plain decimal (`0.1`,
//! `1500.0`) when the decimal exponent is in `-5..16` and in scientific form (`1e21`, `1.5e-7`)
//! otherwise. A float always carries a `.` or an `e`, so `1.0` does not turn into the integer `1` on
//! the way back. NaN and the infinities have no JSON form; the writer refuses them.

use std::fmt::{self, Write as _};

/// How many significant decimal digits of a literal take part in the conversion to `f64`; the rest
/// only set the sticky bit. 800 is above the 767 digits the longest rounding boundary has.
pub const MAX_SIGNIFICANT_DIGITS: usize = 800;

/// A JSON number.
///
/// Parsed numbers are normalised: `Int` for every integer literal that fits `i64`, `UInt` only for
/// integers above `i64::MAX`, `Float` for everything else. Equality compares the two integer forms
/// by value (`Int(5) == UInt(5)`), floats by bit pattern (`0.0 != -0.0`, a NaN equals itself), and an
/// integer never equals a float (`1 != 1.0`), so a value that survives a round trip is equal to the
/// original in type as well as in value.
#[derive(Clone, Copy, Debug)]
pub enum Number {
    /// An integer that fits `i64`.
    Int(i64),
    /// A non-negative integer; the parser uses this only above `i64::MAX`.
    UInt(u64),
    /// Any other number. Finite when it comes from the parser.
    Float(f64),
}

const TWO_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

impl Number {
    /// Parses one JSON number literal (the whole of `text`, no surrounding space). Returns `None`
    /// when the text is not a number by the RFC 8259 grammar or rounds to infinity.
    #[must_use]
    pub fn from_json(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        match scan(bytes, 0) {
            Ok((literal, end)) if end == bytes.len() => convert(&literal).ok(),
            _ => None,
        }
    }

    /// The value as `i64`: an integer in range, or a float that is a whole number in range.
    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(v),
            Self::UInt(v) => i64::try_from(v).ok(),
            Self::Float(f) => (f.fract() == 0.0 && (-TWO_63..TWO_63).contains(&f)).then_some(f as i64),
        }
    }

    /// The value as `u64`: a non-negative integer, or a float that is a whole number in range.
    #[must_use]
    pub fn as_u64(self) -> Option<u64> {
        match self {
            Self::Int(v) => u64::try_from(v).ok(),
            Self::UInt(v) => Some(v),
            Self::Float(f) => (f.fract() == 0.0 && (0.0..TWO_64).contains(&f)).then_some(f as u64),
        }
    }

    /// The value as `f64`. Integers beyond 2^53 are rounded to the nearest float.
    #[must_use]
    pub const fn as_f64(self) -> f64 {
        match self {
            Self::Int(v) => v as f64,
            Self::UInt(v) => v as f64,
            Self::Float(f) => f,
        }
    }

    /// Whether the number is stored as an integer (`Int` or `UInt`).
    #[must_use]
    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::Float(_))
    }

    /// Whether the number can be written as JSON: everything except NaN and the infinities.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        match self {
            Self::Float(f) => f.is_finite(),
            _ => true,
        }
    }

    const fn as_i128(self) -> i128 {
        match self {
            Self::Int(v) => v as i128,
            Self::UInt(v) => v as i128,
            Self::Float(_) => 0,
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Float(_), _) | (_, Self::Float(_)) => false,
            (a, b) => a.as_i128() == b.as_i128(),
        }
    }
}

impl Eq for Number {}

/// Writes the JSON text of the number. A non-finite float, which has no JSON text, is written the
/// way Rust prints it (`NaN`, `inf`); the JSON writer refuses such a number before it gets here.
impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Int(v) => write_i64(f, v),
            Self::UInt(v) => write_u64(f, v),
            Self::Float(v) if v.is_finite() => write_f64(f, v),
            Self::Float(v) => write!(f, "{v}"),
        }
    }
}

macro_rules! number_from_small_int {
    ($($ty:ty),*) => {$(
        impl From<$ty> for Number {
            fn from(v: $ty) -> Self {
                Self::Int(i64::from(v))
            }
        }
    )*};
}

number_from_small_int!(i8, i16, i32, i64, u8, u16, u32);

impl From<u64> for Number {
    fn from(v: u64) -> Self {
        i64::try_from(v).map_or(Self::UInt(v), Self::Int)
    }
}

impl From<usize> for Number {
    fn from(v: usize) -> Self {
        Self::from(v as u64)
    }
}

impl From<isize> for Number {
    fn from(v: isize) -> Self {
        Self::Int(v as i64)
    }
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

/// The float keeps its value exactly (every `f32` is an `f64`); it is then written with the digits
/// of the `f64`, which are longer than the `f32` needs. Use the writer's `f32` method for the short
/// form.
impl From<f32> for Number {
    fn from(v: f32) -> Self {
        Self::Float(f64::from(v))
    }
}

// --- Scanning --------------------------------------------------------------------------------------

/// The parts of a number literal that has been checked against the RFC 8259 grammar.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Literal<'a> {
    negative: bool,
    /// The integer digits (at least one, no superfluous leading zero).
    int: &'a [u8],
    /// The digits after the decimal point; empty when there is no fraction.
    frac: &'a [u8],
    exp_negative: bool,
    /// The exponent digits; empty when there is no exponent.
    exp: &'a [u8],
}

/// Why [`scan`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScanError {
    /// The input ends inside the literal, at this offset.
    Truncated(usize),
    /// The byte at this offset breaks the grammar.
    Invalid(usize, &'static str),
}

/// The value does not fit an `f64` (it rounds to infinity), or, for `TooLong`, the conversion ran
/// out of its fixed work space, which the digit and exponent bounds rule out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvertError {
    /// The magnitude is above the largest finite `f64`.
    Overflow,
    /// The big-integer work space was exceeded.
    TooLong,
}

fn digits(input: &[u8], mut at: usize) -> usize {
    while input.get(at).is_some_and(u8::is_ascii_digit) {
        at += 1;
    }
    at
}

/// Checks the number literal that starts at `start` (`-` or a digit) and returns its parts and the
/// offset just past it.
pub(crate) fn scan(input: &[u8], start: usize) -> Result<(Literal<'_>, usize), ScanError> {
    let mut at = start;
    let negative = input.get(at) == Some(&b'-');
    if negative {
        at += 1;
    }
    let int_start = at;
    match input.get(at) {
        None => return Err(ScanError::Truncated(at)),
        Some(b'0') => {
            at += 1;
            if input.get(at).is_some_and(u8::is_ascii_digit) {
                return Err(ScanError::Invalid(at, "a number has a leading zero"));
            }
        }
        Some(b'1'..=b'9') => at = digits(input, at),
        Some(_) => return Err(ScanError::Invalid(at, "expected a digit")),
    }
    let int = input.get(int_start..at).unwrap_or_default();
    let mut frac: &[u8] = &[];
    if input.get(at) == Some(&b'.') {
        at += 1;
        let end = digits(input, at);
        if end == at {
            return Err(if at >= input.len() {
                ScanError::Truncated(at)
            } else {
                ScanError::Invalid(at, "expected a digit after the decimal point")
            });
        }
        frac = input.get(at..end).unwrap_or_default();
        at = end;
    }
    let mut exp: &[u8] = &[];
    let mut exp_negative = false;
    if matches!(input.get(at), Some(b'e' | b'E')) {
        at += 1;
        match input.get(at) {
            Some(b'-') => {
                exp_negative = true;
                at += 1;
            }
            Some(b'+') => at += 1,
            _ => {}
        }
        let end = digits(input, at);
        if end == at {
            return Err(if at >= input.len() {
                ScanError::Truncated(at)
            } else {
                ScanError::Invalid(at, "expected a digit in the exponent")
            });
        }
        exp = input.get(at..end).unwrap_or_default();
        at = end;
    }
    Ok((Literal { negative, int, frac, exp_negative, exp }, at))
}

// --- Decimal to binary -----------------------------------------------------------------------------

/// Exponents beyond this are clamped; with any literal that fits in memory the result is the same
/// (zero or overflow).
const EXPONENT_CLAMP: i64 = 1 << 40;

/// Powers of ten that are exact in `f64`.
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20,
    1e21, 1e22,
];

/// 5^0 ..= 5^27, the powers of five below 2^63.
const POW5_U64: [u64; 28] = {
    let mut table = [1_u64; 28];
    let mut i = 1;
    while i < 28 {
        table[i] = table[i - 1] * 5;
        i += 1;
    }
    table
};

/// 10^0 ..= 10^19, the powers of ten below 2^64.
const POW10_U64: [u64; 20] = {
    let mut table = [1_u64; 20];
    let mut i = 1;
    while i < 20 {
        table[i] = table[i - 1] * 10;
        i += 1;
    }
    table
};

/// 5^13, the largest power of five below 2^32.
const POW5_13: u32 = 1_220_703_125;

/// Converts a scanned literal to a [`Number`].
pub(crate) fn convert(literal: &Literal<'_>) -> Result<Number, ConvertError> {
    if literal.frac.is_empty() && literal.exp.is_empty() && literal.int.len() <= 20 {
        let mut value = Some(0_u64);
        for &d in literal.int {
            value = value.and_then(|v| v.checked_mul(10)).and_then(|v| v.checked_add(u64::from(d - b'0')));
        }
        match value {
            Some(0) if literal.negative => return Ok(Number::Float(-0.0)),
            Some(v) if !literal.negative => return Ok(Number::from(v)),
            Some(v) if v <= 1 << 63 => return Ok(Number::Int((v as i64).wrapping_neg())),
            _ => {}
        }
    }
    let mut exponent = 0_i64;
    for &d in literal.exp {
        exponent = (exponent * 10 + i64::from(d - b'0')).min(EXPONENT_CLAMP);
    }
    if literal.exp_negative {
        exponent = -exponent;
    }
    to_f64(literal, exponent).map(Number::Float)
}

/// Walks the significant digits of a literal: `keep` receives the first `cap` of them (leading zeros
/// are not significant). Returns how many were kept, the power of ten that scales the kept digits
/// (read as an integer) to the value, and whether a non-zero digit was dropped.
fn scan_digits(literal: &Literal<'_>, exponent: i64, cap: usize, mut keep: impl FnMut(u8)) -> (usize, i64, bool) {
    let mut kept = 0_usize;
    let mut dropped_int = 0_i64;
    let mut used_frac = 0_i64;
    let mut sticky = false;
    for &d in literal.int {
        if kept == 0 && d == b'0' {
            continue;
        }
        if kept < cap {
            keep(d - b'0');
            kept += 1;
        } else {
            dropped_int += 1;
            sticky |= d != b'0';
        }
    }
    for &d in literal.frac {
        if kept == 0 && d == b'0' {
            used_frac += 1;
        } else if kept < cap {
            keep(d - b'0');
            kept += 1;
            used_frac += 1;
        } else {
            sticky |= d != b'0';
        }
    }
    (kept, exponent + dropped_int - used_frac, sticky)
}

fn signed(negative: bool, magnitude: f64) -> f64 {
    if negative { -magnitude } else { magnitude }
}

/// What the magnitude of `digits × 10^e10` (with `kept` digits, the first non-zero) allows.
enum Range {
    /// At least 10^309: above every finite `f64`.
    Overflow,
    /// Below 10^-324, less than half the smallest subnormal: rounds to zero.
    Zero,
    /// Has to be computed.
    Compute,
}

fn range(kept: usize, e10: i64) -> Range {
    // The value lies in [10^(point-1), 10^point), and a dropped tail does not reach 10^point.
    let point = kept as i64 + e10;
    if point > 309 {
        Range::Overflow
    } else if point < -323 {
        Range::Zero
    } else {
        Range::Compute
    }
}

fn to_f64(literal: &Literal<'_>, exponent: i64) -> Result<f64, ConvertError> {
    let negative = literal.negative;
    let mut mantissa = 0_u64;
    let (kept, e10, sticky) = scan_digits(literal, exponent, 19, |d| mantissa = mantissa * 10 + u64::from(d));
    if kept == 0 {
        return Ok(signed(negative, 0.0));
    }
    match range(kept, e10) {
        Range::Overflow => return Err(ConvertError::Overflow),
        Range::Zero => return Ok(signed(negative, 0.0)),
        Range::Compute => {}
    }
    if !sticky {
        return from_u64(mantissa, e10, negative);
    }
    // More than 19 significant digits: build the big integer from up to MAX_SIGNIFICANT_DIGITS.
    let mut big = Big::zero();
    let mut ok = true;
    let mut chunk = 0_u32;
    let mut chunk_len = 0_usize;
    let (kept, e10, sticky) = scan_digits(literal, exponent, MAX_SIGNIFICANT_DIGITS, |d| {
        chunk = chunk * 10 + u32::from(d);
        chunk_len += 1;
        if chunk_len == 9 {
            ok &= big.mul_add(1_000_000_000, chunk);
            chunk = 0;
            chunk_len = 0;
        }
    });
    if chunk_len > 0 {
        ok &= big.mul_add(POW10_U64[chunk_len] as u32, chunk);
    }
    if !ok {
        return Err(ConvertError::TooLong);
    }
    match range(kept, e10) {
        Range::Overflow => Err(ConvertError::Overflow),
        Range::Zero => Ok(signed(negative, 0.0)),
        Range::Compute => scale_and_round(&mut big, e10, sticky, negative),
    }
}

/// `mantissa × 10^e10` for a non-zero mantissa below 10^19 that passed [`range`].
fn from_u64(mantissa: u64, e10: i64, negative: bool) -> Result<f64, ConvertError> {
    // Clinger's fast path: both operands are exact, so the one operation rounds correctly.
    if mantissa <= 1 << 53 && (-22..=22).contains(&e10) {
        let value = mantissa as f64;
        let power = POW10_F64[e10.unsigned_abs() as usize];
        return Ok(signed(negative, if e10 < 0 { value / power } else { value * power }));
    }
    if (0..=19).contains(&e10) {
        // Exact: below 2^64 × 2^64.
        let product = u128::from(mantissa) * u128::from(POW10_U64[e10 as usize]);
        return round_u128(product, 0, false, negative);
    }
    if (-27..0).contains(&e10) {
        // mantissa / 10^n = (mantissa × 2^k / 5^n) × 2^(-k-n); the numerator has its top bit at 127
        // and the divisor is below 2^63, so the quotient has more than 64 significant bits.
        let n = e10.unsigned_abs() as usize;
        let wide = u128::from(mantissa);
        let k = wide.leading_zeros();
        let numerator = wide << k;
        let divisor = u128::from(POW5_U64[n]);
        let quotient = numerator / divisor;
        let sticky = numerator % divisor != 0;
        return round_u128(quotient, -i64::from(k) - n as i64, sticky, negative);
    }
    let mut big = Big::zero();
    big.set_u64(mantissa);
    scale_and_round(&mut big, e10, false, negative)
}

/// Rounds `value × 2^exp2` (plus, when `sticky`, a positive amount below one unit of `value`).
fn round_u128(value: u128, exp2: i64, sticky: bool, negative: bool) -> Result<f64, ConvertError> {
    if value == 0 {
        return Ok(signed(negative, 0.0));
    }
    let lz = value.leading_zeros();
    let aligned = value << lz;
    let top = (aligned >> 64) as u64;
    let low = aligned as u64;
    let msb = i64::from(127 - lz) + exp2;
    round_to_f64(top, msb, sticky || low != 0, negative)
}

/// Rounds half-to-even a positive value given by its 64 leading bits.
///
/// `top` holds the leading bits with the most significant one at bit 63; `msb` is the binary
/// exponent of that bit (the value is in `[2^msb, 2^(msb+1))`); `sticky` says that non-zero bits
/// follow the 64.
fn round_to_f64(top: u64, msb: i64, mut sticky: bool, negative: bool) -> Result<f64, ConvertError> {
    if msb > 1023 {
        return Err(ConvertError::Overflow);
    }
    // How many leading bits the result keeps: 53 for a normal number, fewer for a subnormal.
    let precision = if msb >= -1022 { 53 } else { msb + 1075 };
    if precision < 0 {
        // Below 2^-1075, half the smallest subnormal.
        return Ok(signed(negative, 0.0));
    }
    let precision = precision as u32;
    let mut mantissa = if precision == 0 { 0 } else { top >> (64 - precision) };
    let round_bit = (top >> (63 - precision)) & 1;
    sticky |= top & ((1_u64 << (63 - precision)) - 1) != 0;
    if round_bit == 1 && (sticky || mantissa & 1 == 1) {
        mantissa += 1;
    }
    // A mantissa that rounded up to the next power of two carries into the exponent field by
    // itself, for normal and subnormal numbers alike.
    let bits = if msb >= -1022 { (((msb + 1023) as u64) << 52) + (mantissa - (1 << 52)) } else { mantissa };
    if bits >> 52 >= 0x7FF {
        return Err(ConvertError::Overflow);
    }
    Ok(signed(negative, f64::from_bits(bits)))
}

/// An upper bound of `ceil(n × log2(5))`.
const fn log2_pow5_ceil(n: u64) -> u64 {
    // log2(5) = 2.32192809...; 152170 / 65536 = 2.32193...
    ((n * 152_170) >> 16) + 1
}

/// Rounds `big × 10^e10` (plus a positive amount below one unit of `big` when `sticky`).
fn scale_and_round(big: &mut Big, e10: i64, mut sticky: bool, negative: bool) -> Result<f64, ConvertError> {
    let exp2 = if e10 >= 0 {
        // big × 10^e = (big × 5^e) × 2^e, exactly.
        let mut left = e10;
        while left >= 13 {
            if !big.mul_add(POW5_13, 0) {
                return Err(ConvertError::TooLong);
            }
            left -= 13;
        }
        if left > 0 && !big.mul_add(POW5_U64[left as usize] as u32, 0) {
            return Err(ConvertError::TooLong);
        }
        e10
    } else {
        // big / 10^n = (big × 2^k / 5^n) × 2^(-k-n). k makes the quotient at least 66 bits long, so
        // its 64 leading bits and the remainder decide the rounding.
        let n = e10.unsigned_abs();
        let wanted = 66 + log2_pow5_ceil(n);
        let have = big.bit_len() as u64;
        let k = wanted.saturating_sub(have);
        if !big.shl(k as usize) {
            return Err(ConvertError::TooLong);
        }
        let mut left = n;
        while left >= 13 {
            sticky |= big.div_small(POW5_13) != 0;
            left -= 13;
        }
        if left > 0 {
            sticky |= big.div_small(POW5_U64[left as usize] as u32) != 0;
        }
        -(k as i64) - n as i64
    };
    let length = big.bit_len();
    if length == 0 {
        return Ok(signed(negative, 0.0));
    }
    let (top, low) = big.top64();
    round_to_f64(top, length as i64 - 1 + exp2, sticky || low, negative)
}

/// Limbs of the conversion's big integer. The largest value formed has under 2700 bits (800 digits,
/// or 66 bits more than 5^1123); 96 limbs hold 3072.
const BIG_LIMBS: usize = 96;

/// A fixed-capacity unsigned big integer, little-endian 32-bit limbs.
struct Big {
    limbs: [u32; BIG_LIMBS],
    len: usize,
}

impl Big {
    const fn zero() -> Self {
        Self { limbs: [0; BIG_LIMBS], len: 0 }
    }

    /// Sets the value to `v`.
    fn set_u64(&mut self, v: u64) {
        self.limbs = [0; BIG_LIMBS];
        self.limbs[0] = v as u32;
        self.limbs[1] = (v >> 32) as u32;
        self.len = if v >> 32 != 0 { 2 } else { usize::from(v != 0) };
    }

    /// `self = self × mul + add`. Returns `false` (leaving a truncated value) when it does not fit.
    fn mul_add(&mut self, mul: u32, add: u32) -> bool {
        let mut carry = u64::from(add);
        for limb in &mut self.limbs[..self.len] {
            let v = u64::from(*limb) * u64::from(mul) + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            if self.len == BIG_LIMBS {
                return false;
            }
            self.limbs[self.len] = carry as u32;
            self.len += 1;
        }
        true
    }

    /// `self <<= bits`. Returns `false` when the result does not fit.
    fn shl(&mut self, bits: usize) -> bool {
        if self.len == 0 || bits == 0 {
            return true;
        }
        let (whole, part) = (bits / 32, (bits % 32) as u32);
        let new_len = self.len + whole + 1;
        if new_len > BIG_LIMBS {
            return false;
        }
        for i in (0..new_len).rev() {
            // The limb that lands at i comes from i - whole and, for a partial shift, i - whole - 1.
            let high = if i >= whole && i - whole < self.len { self.limbs[i - whole] } else { 0 };
            let low = if i > whole && i - whole - 1 < self.len { self.limbs[i - whole - 1] } else { 0 };
            self.limbs[i] = if part == 0 { high } else { (high << part) | (low >> (32 - part)) };
        }
        self.len = new_len;
        self.trim();
        true
    }

    /// `self /= divisor` (which must be non-zero); returns the remainder.
    fn div_small(&mut self, divisor: u32) -> u32 {
        if divisor == 0 {
            return 0;
        }
        let mut rem = 0_u64;
        for limb in self.limbs[..self.len].iter_mut().rev() {
            let cur = (rem << 32) | u64::from(*limb);
            *limb = (cur / u64::from(divisor)) as u32;
            rem = cur % u64::from(divisor);
        }
        self.trim();
        rem as u32
    }

    fn trim(&mut self) {
        while self.len > 0 && self.limbs[self.len - 1] == 0 {
            self.len -= 1;
        }
    }

    /// The number of bits up to and including the highest set one.
    fn bit_len(&self) -> usize {
        if self.len == 0 { 0 } else { self.len * 32 - self.limbs[self.len - 1].leading_zeros() as usize }
    }

    fn limb(&self, i: usize) -> u128 {
        if i < self.len { u128::from(self.limbs[i]) } else { 0 }
    }

    /// The 64 leading bits, most significant at bit 63, and whether any lower bit is set.
    fn top64(&self) -> (u64, bool) {
        let length = self.bit_len();
        if length <= 64 {
            let value = (self.limb(0) | (self.limb(1) << 32)) as u64;
            return (if length == 0 { 0 } else { value << (64 - length) }, false);
        }
        let shift = length - 64;
        let (first, offset) = (shift / 32, shift % 32);
        let window = self.limb(first) | (self.limb(first + 1) << 32) | (self.limb(first + 2) << 64);
        let top = (window >> offset) as u64;
        let below = self.limbs[..first].iter().any(|&l| l != 0) || window & ((1_u128 << offset) - 1) != 0;
        (top, below)
    }
}

// --- Binary to decimal -----------------------------------------------------------------------------

/// A small text buffer on the stack, for formatting one number.
struct StackText {
    bytes: [u8; 40],
    len: usize,
}

impl StackText {
    const fn new() -> Self {
        Self { bytes: [0; 40], len: 0 }
    }

    fn as_str(&self) -> Result<&str, fmt::Error> {
        std::str::from_utf8(&self.bytes[..self.len]).map_err(|_| fmt::Error)
    }
}

impl fmt::Write for StackText {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        let slot = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        slot.copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

const ZEROS: &str = "0000000000000000";

fn write_zeros<W: fmt::Write>(out: &mut W, mut count: usize) -> fmt::Result {
    while count > 0 {
        let n = count.min(ZEROS.len());
        out.write_str(&ZEROS[..n])?;
        count -= n;
    }
    Ok(())
}

/// Lays out shortest digits (`{:e}` text: `d[.ddd]e[-]x`, sign already removed) as JSON.
fn layout<W: fmt::Write>(out: &mut W, text: &str) -> fmt::Result {
    let (mantissa, exponent_text) = text.split_once('e').ok_or(fmt::Error)?;
    let exponent: i32 = exponent_text.parse().map_err(|_| fmt::Error)?;
    let (first, rest) = mantissa.split_at_checked(1).ok_or(fmt::Error)?;
    let rest = rest.strip_prefix('.').unwrap_or(rest);
    if (0..16).contains(&exponent) {
        // d ddd . ddd, padded with zeros up to the decimal point.
        let int_more = exponent as usize;
        out.write_str(first)?;
        if rest.len() > int_more {
            out.write_str(&rest[..int_more])?;
            out.write_char('.')?;
            out.write_str(&rest[int_more..])
        } else {
            out.write_str(rest)?;
            write_zeros(out, int_more - rest.len())?;
            out.write_str(".0")
        }
    } else if (-5..0).contains(&exponent) {
        out.write_str("0.")?;
        write_zeros(out, (-exponent - 1) as usize)?;
        out.write_str(first)?;
        out.write_str(rest)
    } else {
        out.write_str(first)?;
        if !rest.is_empty() {
            out.write_char('.')?;
            out.write_str(rest)?;
        }
        out.write_char('e')?;
        out.write_str(exponent_text)
    }
}

/// Writes a finite `f64` with the shortest digits that read back as the same value (see the module
/// comment). The caller has ruled out NaN and the infinities.
pub(crate) fn write_f64<W: fmt::Write>(out: &mut W, value: f64) -> fmt::Result {
    if value.is_sign_negative() {
        out.write_char('-')?;
    }
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return out.write_str("0.0");
    }
    let mut text = StackText::new();
    write!(text, "{magnitude:e}")?;
    layout(out, text.as_str()?)
}

/// Writes a finite `f32` so that reading the text as `f64` and narrowing it gives back `value`.
///
/// The shortest digits of the `f32` are used when that holds, which it does except for decimals that
/// sit within an `f64` rounding error of the midpoint of two `f32` values; then the digits of the
/// widened value are written instead.
pub(crate) fn write_f32<W: fmt::Write>(out: &mut W, value: f32) -> fmt::Result {
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return write_f64(out, f64::from(value));
    }
    let mut text = StackText::new();
    write!(text, "{magnitude:e}")?;
    let mut json = StackText::new();
    layout(&mut json, text.as_str()?)?;
    let back = Number::from_json(json.as_str()?).map(|n| n.as_f64() as f32);
    if back != Some(magnitude) {
        return write_f64(out, f64::from(value));
    }
    if value.is_sign_negative() {
        out.write_char('-')?;
    }
    out.write_str(json.as_str()?)
}

/// Writes an unsigned integer in decimal.
pub(crate) fn write_u64<W: fmt::Write>(out: &mut W, mut value: u64) -> fmt::Result {
    let mut buf = [0_u8; 20];
    let mut at = buf.len();
    loop {
        at -= 1;
        buf[at] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    out.write_str(std::str::from_utf8(&buf[at..]).map_err(|_| fmt::Error)?)
}

/// Writes a signed integer in decimal.
pub(crate) fn write_i64<W: fmt::Write>(out: &mut W, value: i64) -> fmt::Result {
    if value < 0 {
        out.write_char('-')?;
    }
    write_u64(out, value.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn float(text: &str) -> f64 {
        match Number::from_json(text) {
            Some(Number::Float(f)) => f,
            other => panic!("{text}: expected a float, got {other:?}"),
        }
    }

    fn written(value: f64) -> String {
        let mut s = String::new();
        write_f64(&mut s, value).unwrap();
        s
    }

    #[test]
    fn integers_keep_every_bit() {
        assert_eq!(Number::from_json("0"), Some(Number::Int(0)));
        assert_eq!(Number::from_json("-1"), Some(Number::Int(-1)));
        assert_eq!(Number::from_json("9223372036854775807"), Some(Number::Int(i64::MAX)));
        assert_eq!(Number::from_json("-9223372036854775808"), Some(Number::Int(i64::MIN)));
        assert!(matches!(Number::from_json("9223372036854775808"), Some(Number::UInt(9_223_372_036_854_775_808))));
        assert!(matches!(Number::from_json("18446744073709551615"), Some(Number::UInt(u64::MAX))));
        // 2^53 + 1 is not an f64, but it is an i64.
        assert_eq!(Number::from_json("9007199254740993"), Some(Number::Int(9_007_199_254_740_993)));
        // One past either end becomes the nearest float.
        assert_eq!(float("18446744073709551616"), 18_446_744_073_709_551_616.0);
        assert_eq!(float("-9223372036854775809"), -9_223_372_036_854_775_808.0);
        assert_eq!(float("123456789012345678901234567890"), 123_456_789_012_345_678_901_234_567_890.0);
        // -0 keeps its sign, as a float.
        let z = float("-0");
        assert!(z == 0.0 && z.is_sign_negative());
        assert_eq!(float("-0.0").to_bits(), (-0.0_f64).to_bits());
        assert_eq!(float("0e0").to_bits(), 0);
        assert_eq!(float("0.0").to_bits(), 0);
        assert_eq!(float("1.0"), 1.0);
        assert_eq!(float("1e0"), 1.0);
    }

    #[test]
    fn the_grammar_is_strict() {
        for bad in [
            "",
            "-",
            "+1",
            "01",
            "-01",
            "00",
            "1.",
            ".5",
            "1.e3",
            "1e",
            "1e+",
            "1e-",
            "1E",
            "0x10",
            "1 ",
            " 1",
            "1a",
            "--1",
            "1..2",
            "1.2.3",
            "1e2e3",
            "Infinity",
            "NaN",
            "-Infinity",
            "1_000",
            "１",
        ] {
            assert_eq!(Number::from_json(bad), None, "{bad:?}");
        }
        for good in
            ["0", "-0", "10", "1.5", "1e5", "1E5", "1e+5", "1e-5", "0.0e-0", "123.456E+7", "0e99999999999999999999"]
        {
            assert!(Number::from_json(good).is_some(), "{good:?}");
        }
        assert_eq!(scan(b"1.", 0).unwrap_err(), ScanError::Truncated(2));
        assert_eq!(scan(b"-", 0).unwrap_err(), ScanError::Truncated(1));
        assert_eq!(scan(b"1e", 0).unwrap_err(), ScanError::Truncated(2));
        assert_eq!(scan(b"1e+", 0).unwrap_err(), ScanError::Truncated(3));
        assert!(matches!(scan(b"012", 0), Err(ScanError::Invalid(1, _))));
        assert!(matches!(scan(b"-x", 0), Err(ScanError::Invalid(1, _))));
        assert!(matches!(scan(b"1.x", 0), Err(ScanError::Invalid(2, _))));
        assert!(matches!(scan(b"1ex", 0), Err(ScanError::Invalid(2, _))));
        // A literal ends where the grammar ends; what follows is the caller's business.
        assert_eq!(scan(b"[12,3]", 1).unwrap().1, 3);
    }

    #[test]
    fn known_hard_cases_round_correctly() {
        // The largest subnormal, the literal that hung PHP and Java in 2011.
        assert_eq!(float("2.2250738585072011e-308").to_bits(), 0x000F_FFFF_FFFF_FFFF);
        assert_eq!(float("2.2250738585072012e-308").to_bits(), 0x0010_0000_0000_0000);
        assert_eq!(float("2.2250738585072014e-308"), f64::MIN_POSITIVE);
        // The smallest subnormal and the halfway point below it (2^-1075 = 2.4703282292062327208...e-324).
        assert_eq!(float("5e-324").to_bits(), 1);
        assert_eq!(float("4.9e-324").to_bits(), 1);
        assert_eq!(float("4.9406564584124654e-324").to_bits(), 1);
        assert_eq!(float("2.4703282292062328e-324").to_bits(), 1);
        assert_eq!(float("2.4703282292062327e-324").to_bits(), 0);
        assert_eq!(float("-2.4703282292062327e-324").to_bits(), 1 << 63);
        assert_eq!(float("1e-400").to_bits(), 0);
        assert_eq!(float("-1e-400").to_bits(), 1 << 63);
        assert_eq!(float("1e-99999999999999999999").to_bits(), 0);
        // The largest finite value and the halfway point above it (1.797693134862315807...e308).
        assert_eq!(float("1.7976931348623157e308"), f64::MAX);
        assert_eq!(float("1.7976931348623158e308"), f64::MAX);
        assert_eq!(
            float(
                "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791.9999999999999999999999999999999999999999999999999999999999999999999999"
            ),
            f64::MAX
        );
        assert_eq!(
            Number::from_json(
                "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792"
            ),
            None
        );
        assert_eq!(Number::from_json("1.7976931348623159e308"), None);
        assert_eq!(Number::from_json("1e309"), None);
        assert_eq!(Number::from_json("1e400"), None);
        assert_eq!(Number::from_json("-1e400"), None);
        assert_eq!(Number::from_json("1e99999999999999999999"), None);
        // Everyday values.
        assert_eq!(float("0.1"), 0.1);
        assert_eq!(float("0.2"), 0.2);
        assert_eq!(float("0.30000000000000004"), 0.1 + 0.2);
        assert_eq!(float("0.3"), 0.3);
        assert_ne!(float("0.3"), 0.1 + 0.2);
        assert_eq!(float("3.141592653589793"), std::f64::consts::PI);
        assert_eq!(float("1e23"), 1e23);
        assert_eq!(float("8.5e-7"), 8.5e-7);
        assert_eq!(float("6.02214076e23"), 6.022_140_76e23);
        // Ties between two floats go to the even one: 2^53 + 1 and 2^53 + 3 as floats.
        assert_eq!(float("9007199254740993.0"), 9_007_199_254_740_992.0);
        assert_eq!(float("9007199254740995.0"), 9_007_199_254_740_996.0);
        assert_eq!(float("9007199254740993.00000000000000000000000000000000000001"), 9_007_199_254_740_994.0);
        assert_eq!(float("9.007199254740993e15"), 9_007_199_254_740_992.0);
        // Published stress cases for decimal-to-double conversion.
        assert_eq!(float("9214843084008499.0"), 9_214_843_084_008_499.0);
        assert_eq!(float("0.500000000000000166533453693773481063544750213623046875"), 0.500_000_000_000_000_2);
        assert_eq!(float("3.0540412E5"), 305_404.12);
        assert_eq!(float("8.98846567431158e307"), 8.988_465_674_311_58e307);
        assert_eq!(float("7.2057594037927933e16"), 72_057_594_037_927_936.0);
        assert_eq!(float("1.00000000000000011102230246251565404236316680908203125"), 1.0);
        assert_eq!(float("1.00000000000000011102230246251565404236316680908203126"), 1.000_000_000_000_000_2);
        assert_eq!(float("1.00000000000000033306690738754696212708950042724609375"), 1.000_000_000_000_000_4);
    }

    /// The exact decimal expansion of a non-negative float, as an integer scaled by 10^1100.
    fn exact_digits(value: f64) -> Vec<u8> {
        let text = format!("{value:.1100}");
        text.bytes().filter(u8::is_ascii_digit).map(|d| d - b'0').collect()
    }

    fn add_digits(a: &[u8], b: &[u8]) -> Vec<u8> {
        let n = a.len().max(b.len());
        let mut out = vec![0_u8; n + 1];
        let mut carry = 0;
        for i in 0..n {
            let x = if i < a.len() { a[a.len() - 1 - i] } else { 0 };
            let y = if i < b.len() { b[b.len() - 1 - i] } else { 0 };
            let s = x + y + carry;
            out[n - i] = s % 10;
            carry = s / 10;
        }
        out[0] = carry;
        out
    }

    fn halve_digits(a: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(a.len() + 1);
        let mut rem = 0;
        for &d in a {
            let cur = rem * 10 + d;
            out.push(cur / 2);
            rem = cur % 2;
        }
        out.push(rem * 5);
        out
    }

    /// Scaled digits (an integer times 10^-scale) as a JSON literal.
    fn literal(digits: &[u8], scale: usize) -> String {
        let text: String = digits.iter().map(|d| char::from(b'0' + d)).collect();
        let split = text.len() - scale;
        let int = text[..split].trim_start_matches('0');
        format!("{}.{}", if int.is_empty() { "0" } else { int }, &text[split..])
    }

    #[test]
    fn exact_halfway_literals_round_to_even_and_a_hair_more_rounds_up() {
        // For a and its successor b, the literal of (a + b) / 2 is an exact tie; it has up to 767
        // significant digits near the bottom of the range. One more digit anywhere after it, even
        // beyond the digits the converter keeps, must tip it upwards.
        let mut rng = Rng::new(0x7EA5);
        let mut cases: Vec<f64> = vec![
            0.0,
            f64::from_bits(1),
            f64::from_bits(2),
            f64::from_bits(0x000F_FFFF_FFFF_FFFE),
            f64::from_bits(0x000F_FFFF_FFFF_FFFF),
            f64::MIN_POSITIVE,
            1.0,
            f64::from_bits(1.0_f64.to_bits() - 1),
            0.1,
            1e-5,
            1e-300,
            123.456,
        ];
        for _ in 0..300 {
            // Subnormals and the lowest binades, where the expansions are longest.
            cases.push(f64::from_bits(rng.next_u64() >> rng.range_i64(1, 12)));
        }
        for _ in 0..300 {
            let bits = rng.next_u64() & 0x7FFF_FFFF_FFFF_FFFF;
            let f = f64::from_bits(bits);
            if f.is_finite() && f < 1e30 {
                cases.push(f);
            }
        }
        for a in cases {
            let b = f64::from_bits(a.to_bits() + 1);
            if !b.is_finite() {
                continue;
            }
            // 1101 fractional digits after the halving.
            let middle = halve_digits(&add_digits(&exact_digits(a), &exact_digits(b)));
            let tie = literal(&middle, 1101);
            let even = if a.to_bits() & 1 == 0 { a } else { b };
            assert_eq!(float(&tie).to_bits(), even.to_bits(), "tie above {a:e}");
            assert_eq!(tie.parse::<f64>().unwrap().to_bits(), even.to_bits(), "std disagrees on {a:e}");
            let above = format!("{tie}1");
            assert_eq!(float(&above).to_bits(), b.to_bits(), "just above the tie over {a:e}");
            let far_above = format!("{tie}{}1", "0".repeat(300));
            assert_eq!(float(&far_above).to_bits(), b.to_bits(), "sticky digit over {a:e}");
            // Just below: the last non-zero digit of the tie lowered by one, then nines.
            let mut below = middle.clone();
            let last = below.iter().rposition(|&d| d != 0).unwrap();
            below[last] -= 1;
            below.extend_from_slice(&[9; 40]);
            let below = literal(&below, 1101 + 40);
            assert_eq!(float(&below).to_bits(), a.to_bits(), "just below the tie over {a:e}");
        }
    }

    fn random_literal(rng: &mut Rng) -> String {
        let mut s = String::new();
        if rng.chance(0.3) {
            s.push('-');
        }
        let int_len = match rng.index(10) {
            0 => 0,
            1 => 20 + rng.index(30),
            2 => 300 + rng.index(600),
            _ => 1 + rng.index(19),
        };
        if int_len == 0 {
            s.push('0');
        } else {
            s.push(char::from(b'1' + rng.index(9) as u8));
            for _ in 1..int_len {
                s.push(char::from(b'0' + rng.index(10) as u8));
            }
        }
        let with_frac = rng.chance(0.6);
        if with_frac {
            s.push('.');
            let frac_len = match rng.index(10) {
                0 => 30 + rng.index(60),
                1 => 400 + rng.index(700),
                _ => 1 + rng.index(20),
            };
            let zeros = if rng.chance(0.3) { rng.index(frac_len) } else { 0 };
            for i in 0..frac_len {
                s.push(if i < zeros { '0' } else { char::from(b'0' + rng.index(10) as u8) });
            }
        }
        if rng.chance(0.6) || !with_frac {
            s.push(if rng.chance(0.5) { 'e' } else { 'E' });
            let exp = rng.range_i64(-360, 330) - if int_len > 20 { int_len as i64 / 2 } else { 0 };
            if exp >= 0 && rng.chance(0.3) {
                s.push('+');
            }
            s.push_str(&exp.to_string());
        }
        s
    }

    #[test]
    fn random_literals_agree_with_the_standard_library() {
        // The standard library's parser is correctly rounded and shares no code with this one.
        let mut rng = Rng::new(0xD0_0B1E);
        let (mut finite, mut overflow, mut zero) = (0, 0, 0);
        for _ in 0..60_000 {
            let text = random_literal(&mut rng);
            let expected: f64 = text.parse().unwrap();
            match Number::from_json(&text) {
                Some(Number::Float(f)) => {
                    assert_eq!(f.to_bits(), expected.to_bits(), "{text}");
                    finite += 1;
                    zero += u32::from(f == 0.0);
                }
                None => {
                    assert!(expected.is_infinite(), "{text} was refused, std reads {expected:e}");
                    overflow += 1;
                }
                other => panic!("{text}: {other:?}"),
            }
        }
        assert!(finite > 30_000 && overflow > 500 && zero > 500, "{finite} {overflow} {zero}");
    }

    #[test]
    fn random_short_literals_agree_with_the_standard_library() {
        // The fast paths: up to 19 digits with exponents around their limits (22, 19, -27).
        let mut rng = Rng::new(0xFA57);
        for _ in 0..200_000 {
            let digits = 1 + rng.index(19);
            let mantissa = rng.below(POW10_U64[digits]);
            let exp = rng.range_i64(-45, 45);
            let text = format!("{mantissa}e{exp}");
            let expected: f64 = text.parse().unwrap();
            assert_eq!(float(&text).to_bits(), expected.to_bits(), "{text}");
            let shifted = format!("{}.{:05}e{exp}", mantissa, rng.below(100_000));
            let expected: f64 = shifted.parse().unwrap();
            assert_eq!(float(&shifted).to_bits(), expected.to_bits(), "{shifted}");
        }
    }

    #[test]
    fn random_f64_bit_patterns_round_trip_exactly() {
        // [AC-WP03-1] 300 000 random bit patterns: write, read, same bits.
        let mut rng = Rng::new(0xB175);
        let mut text = String::new();
        let mut checked = 0;
        while checked < 300_000 {
            let bits = rng.next_u64();
            let value = f64::from_bits(bits);
            if !value.is_finite() {
                continue;
            }
            text.clear();
            write_f64(&mut text, value).unwrap();
            assert_eq!(float(&text).to_bits(), bits, "{text}");
            // The standard library's own shortest forms read back the same way.
            assert_eq!(float(&format!("{value:e}")).to_bits(), bits);
            checked += 1;
        }
        // Subnormals and values near one, which uniform bit patterns rarely hit.
        for _ in 0..50_000 {
            let sub = f64::from_bits(rng.next_u64() >> 12);
            text.clear();
            write_f64(&mut text, sub).unwrap();
            assert_eq!(float(&text).to_bits(), sub.to_bits(), "{text}");
            let near = rng.range_f64(-1000.0, 1000.0);
            text.clear();
            write_f64(&mut text, near).unwrap();
            assert_eq!(float(&text).to_bits(), near.to_bits(), "{text}");
        }
    }

    #[test]
    fn random_f32_bit_patterns_round_trip_through_f64() {
        let mut rng = Rng::new(0xF32);
        let mut text = String::new();
        let mut short = 0;
        for _ in 0..300_000 {
            let value = f32::from_bits(rng.next_u32());
            if !value.is_finite() {
                continue;
            }
            text.clear();
            write_f32(&mut text, value).unwrap();
            let back = float(&text) as f32;
            assert_eq!(back.to_bits(), value.to_bits(), "{text}");
            short += u32::from(text.len() <= 16);
        }
        assert!(short > 250_000, "the short form is the normal case: {short}");
        let mut s = String::new();
        write_f32(&mut s, 0.1).unwrap();
        assert_eq!(s, "0.1");
        s.clear();
        write_f32(&mut s, -0.0).unwrap();
        assert_eq!(s, "-0.0");
        s.clear();
        write_f32(&mut s, 16_777_216.0).unwrap();
        assert_eq!(s, "16777216.0");
    }

    #[test]
    fn floats_are_laid_out_readably() {
        assert_eq!(written(0.0), "0.0");
        assert_eq!(written(-0.0), "-0.0");
        assert_eq!(written(1.0), "1.0");
        assert_eq!(written(-1.5), "-1.5");
        assert_eq!(written(0.1), "0.1");
        assert_eq!(written(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(written(123_456.789), "123456.789");
        assert_eq!(written(1500.0), "1500.0");
        assert_eq!(written(1e15), "1000000000000000.0");
        assert_eq!(written(9_007_199_254_740_992.0), "9007199254740992.0");
        assert_eq!(written(1e16), "1e16");
        assert_eq!(written(1.5e16), "1.5e16");
        assert_eq!(written(1e21), "1e21");
        assert_eq!(written(0.000_01), "0.00001");
        assert_eq!(written(0.000_012_5), "0.0000125");
        assert_eq!(written(0.000_001), "1e-6");
        assert_eq!(written(1.5e-7), "1.5e-7");
        assert_eq!(written(5e-324), "5e-324");
        assert_eq!(written(f64::MAX), "1.7976931348623157e308");
        assert_eq!(written(f64::MIN_POSITIVE), "2.2250738585072014e-308");
        assert_eq!(written(-f64::MAX), "-1.7976931348623157e308");
        // Every float form carries a '.' or an 'e', so it reads back as a float.
        for v in [1.0, 100.0, 1e15, 1e16, 1e300, 2.0_f64.powi(60)] {
            assert!(matches!(Number::from_json(&written(v)), Some(Number::Float(f)) if f == v), "{v}");
        }
    }

    #[test]
    fn integers_are_written_in_full() {
        let mut s = String::new();
        write_i64(&mut s, i64::MIN).unwrap();
        s.push(' ');
        write_i64(&mut s, i64::MAX).unwrap();
        s.push(' ');
        write_u64(&mut s, u64::MAX).unwrap();
        s.push(' ');
        write_u64(&mut s, 0).unwrap();
        s.push(' ');
        write_i64(&mut s, -7).unwrap();
        assert_eq!(s, "-9223372036854775808 9223372036854775807 18446744073709551615 0 -7");
        assert_eq!(Number::Int(-12).to_string(), "-12");
        assert_eq!(Number::UInt(12).to_string(), "12");
        assert_eq!(Number::Float(2.5).to_string(), "2.5");
        assert_eq!(Number::Float(f64::NAN).to_string(), "NaN");
        let mut rng = Rng::new(3);
        for _ in 0..20_000 {
            let v = rng.next_u64() as i64 >> rng.index(64);
            assert_eq!(Number::Int(v).to_string(), v.to_string());
            assert_eq!(Number::from_json(&v.to_string()), Some(Number::Int(v)));
        }
    }

    #[test]
    fn equality_and_conversions() {
        assert_eq!(Number::Int(5), Number::UInt(5));
        assert_ne!(Number::Int(-1), Number::UInt(u64::MAX));
        assert_ne!(Number::Int(1), Number::Float(1.0));
        assert_ne!(Number::Float(0.0), Number::Float(-0.0));
        assert_eq!(Number::Float(f64::NAN), Number::Float(f64::NAN));
        assert_eq!(Number::from(7_u64), Number::Int(7));
        assert!(matches!(Number::from(u64::MAX), Number::UInt(u64::MAX)));
        assert_eq!(Number::from(7_usize), Number::Int(7));
        assert_eq!(Number::from(-7_isize), Number::Int(-7));
        assert_eq!(Number::from(-3_i8), Number::Int(-3));
        assert_eq!(Number::from(3_u16), Number::Int(3));
        assert_eq!(Number::from(0.5_f32), Number::Float(0.5));
        assert_eq!(Number::from(0.25_f64), Number::Float(0.25));

        assert_eq!(Number::Int(-3).as_i64(), Some(-3));
        assert_eq!(Number::Int(-3).as_u64(), None);
        assert_eq!(Number::UInt(u64::MAX).as_i64(), None);
        assert_eq!(Number::UInt(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(Number::Float(4.0).as_i64(), Some(4));
        assert_eq!(Number::Float(4.0).as_u64(), Some(4));
        assert_eq!(Number::Float(-4.0).as_u64(), None);
        assert_eq!(Number::Float(4.5).as_i64(), None);
        assert_eq!(Number::Float(4.5).as_u64(), None);
        assert_eq!(Number::Float(f64::NAN).as_i64(), None);
        assert_eq!(Number::Float(f64::INFINITY).as_u64(), None);
        assert_eq!(Number::Float(TWO_63).as_i64(), None);
        assert_eq!(Number::Float(-TWO_63).as_i64(), Some(i64::MIN));
        assert_eq!(Number::Float(TWO_64).as_u64(), None);
        assert_eq!(Number::Float(TWO_63).as_u64(), Some(1 << 63));
        assert_eq!(Number::Int(3).as_f64(), 3.0);
        assert_eq!(Number::UInt(3).as_f64(), 3.0);
        assert!(Number::Int(3).is_integer() && Number::UInt(3).is_integer() && !Number::Float(3.0).is_integer());
        assert!(Number::Int(3).is_finite() && Number::Float(3.0).is_finite());
        assert!(!Number::Float(f64::NAN).is_finite() && !Number::Float(f64::NEG_INFINITY).is_finite());
    }

    #[test]
    fn the_big_integer_does_its_arithmetic() {
        let mut big = Big::zero();
        assert_eq!(big.bit_len(), 0);
        assert_eq!(big.top64(), (0, false));
        big.set_u64(0x1_0000_0001);
        assert_eq!(big.bit_len(), 33);
        assert!(big.shl(95));
        assert_eq!(big.bit_len(), 128);
        assert_eq!(big.top64(), (0x8000_0000_8000_0000, false));
        assert!(big.mul_add(10, 7));
        assert_eq!(big.div_small(10), 7);
        assert_eq!(big.bit_len(), 128);
        assert!(big.mul_add(1, 1));
        assert_eq!(big.top64(), (0x8000_0000_8000_0000, true));
        // Capacity is reported, not overrun.
        assert!(!big.shl(BIG_LIMBS * 32));
        let mut full = Big::zero();
        full.set_u64(u64::MAX);
        let mut fits = true;
        for _ in 0..BIG_LIMBS {
            fits &= full.mul_add(u32::MAX, u32::MAX);
        }
        assert!(!fits);
        assert_eq!(big.div_small(0), 0);
        // A value in u128 survives a shift, a multiplication and a division.
        let mut rng = Rng::new(9);
        for _ in 0..2000 {
            let v = rng.next_u64();
            // 64 + 30 + 32 bits: the reference value stays inside u128.
            let shift = rng.index(31);
            let m = rng.next_u32() | 1;
            let mut big = Big::zero();
            big.set_u64(v);
            assert!(big.shl(shift));
            assert!(big.mul_add(m, 5));
            let expected = (u128::from(v) << shift) * u128::from(m) + 5;
            assert_eq!(big.bit_len() as u32, 128 - expected.leading_zeros());
            let d = (rng.next_u32() >> rng.index(31)) | 1;
            assert_eq!(u128::from(big.div_small(d)), expected % u128::from(d));
            let q = expected / u128::from(d);
            let got = (0..4).fold(0_u128, |acc, i| acc | (big.limb(i) << (32 * i)));
            assert_eq!(got, q);
        }
    }
}
