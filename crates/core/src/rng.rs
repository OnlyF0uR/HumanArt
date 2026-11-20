//! The in-house pseudo-random number generator `[DEP-05]`.
//!
//! `Rng` is xoshiro256++ seeded through SplitMix64: fast, 256 bits of state, and the same sequence on
//! every platform, which is what sampling `[ID-06]`, the fuzz tests `[ROB-01]` and the property tests
//! (section 15.1) rely on. It is not cryptographic: the agent server's bearer token is derived
//! elsewhere, from operating-system entropy.

/// One step of SplitMix64: advances `state` and returns the next value.
///
/// Used to expand a 64-bit seed into the generator's 256-bit state, and useful on its own as a cheap
/// integer mixer.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic xoshiro256++ generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// A generator whose whole sequence is decided by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Self { s }
    }

    /// A generator for an independent stream: the same `seed` with another `stream` gives a sequence
    /// unrelated to the first (per-strand and per-job randomness).
    #[must_use]
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut mix = stream;
        Self::new(seed ^ splitmix64(&mut mix).rotate_left(17))
    }

    /// The next 64 uniformly random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// The next 32 uniformly random bits (the high half, which mixes best).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A uniform `f32` in `[0, 1)` with 24 random bits.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// A uniform `f64` in `[0, 1)` with 53 random bits.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A uniform `f32` in `[lo, hi)`. Returns `lo` when the range is empty or not finite.
    #[inline]
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let v = lo + (hi - lo) * self.f32();
        if v.is_finite() && v < hi { v } else { lo }
    }

    /// A uniform `f64` in `[lo, hi)`. Returns `lo` when the range is empty or not finite.
    #[inline]
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let v = lo + (hi - lo) * self.f64();
        if v.is_finite() && v < hi { v } else { lo }
    }

    /// A uniform integer in `[0, n)` without modulo bias (Lemire's method). Returns 0 when `n` is 0.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as u64
    }

    /// A uniform index in `[0, len)`. Returns 0 when `len` is 0.
    #[inline]
    pub fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// A uniform integer in `[lo, hi]` (both ends included). Returns `lo` when `hi < lo`.
    #[inline]
    pub fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = hi.wrapping_sub(lo) as u64;
        if span == u64::MAX { self.next_u64() as i64 } else { lo.wrapping_add(self.below(span + 1) as i64) }
    }

    /// `true` with probability `p` (clamped to `[0, 1]`).
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// A standard normal deviate (mean 0, variance 1), by the Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        // 1 - f64() is in (0, 1], so the logarithm is finite.
        let u = 1.0 - self.f64();
        let v = self.f64();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// A standard normal deviate as `f32`.
    pub fn normal_f32(&mut self) -> f32 {
        self.normal() as f32
    }

    /// Fills `out` with random bytes.
    pub fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    /// Shuffles `items` in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference values of xoshiro256++ for the state {1, 2, 3, 4}, from the authors' C code.
    #[test]
    fn matches_reference_sequence() {
        let mut rng = Rng { s: [1, 2, 3, 4] };
        let expect = [41_943_041_u64, 58_720_359, 3_588_806_011_781_223, 3_591_011_842_654_386];
        for e in expect {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn same_seed_same_sequence_and_streams_differ() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        let mut c = Rng::with_stream(7, 1);
        let mut same_as_c = 0;
        for _ in 0..1000 {
            let x = a.next_u64();
            assert_eq!(x, b.next_u64());
            same_as_c += u32::from(x == c.next_u64());
        }
        assert_eq!(same_as_c, 0);
    }

    #[test]
    fn floats_stay_in_range_and_cover_it() {
        let mut rng = Rng::new(1);
        let (mut lo, mut hi, mut sum) = (1.0_f64, 0.0_f64, 0.0_f64);
        for _ in 0..100_000 {
            let f = rng.f32();
            assert!((0.0..1.0).contains(&f));
            let d = rng.f64();
            assert!((0.0..1.0).contains(&d));
            lo = lo.min(d);
            hi = hi.max(d);
            sum += d;
        }
        assert!(lo < 0.001 && hi > 0.999);
        assert!((sum / 100_000.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn below_is_unbiased_enough_and_handles_edges() {
        let mut rng = Rng::new(3);
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
        let mut counts = [0_u32; 7];
        for _ in 0..70_000 {
            counts[rng.below(7) as usize] += 1;
        }
        for c in counts {
            assert!((9_500..10_500).contains(&c), "{counts:?}");
        }
        for _ in 0..1000 {
            let v = rng.range_i64(-3, 3);
            assert!((-3..=3).contains(&v));
        }
        assert_eq!(rng.range_i64(5, 5), 5);
        assert_eq!(rng.range_i64(5, 1), 5);
        let _ = rng.range_i64(i64::MIN, i64::MAX);
    }

    #[test]
    fn normal_has_unit_variance() {
        let mut rng = Rng::new(11);
        let n = 200_000;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..n {
            let x = rng.normal();
            assert!(x.is_finite());
            sum += x;
            sq += x * x;
        }
        let mean = sum / f64::from(n);
        let var = sq / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.02, "{var}");
    }

    #[test]
    fn shuffle_keeps_every_item_and_fill_covers_tails() {
        let mut rng = Rng::new(5);
        let mut items: Vec<u32> = (0..100).collect();
        rng.shuffle(&mut items);
        assert_ne!(items, (0..100).collect::<Vec<_>>());
        items.sort_unstable();
        assert_eq!(items, (0..100).collect::<Vec<_>>());
        let mut bytes = [0_u8; 13];
        rng.fill(&mut bytes);
        assert!(bytes.iter().any(|&b| b != 0));
    }

    #[test]
    fn empty_and_broken_ranges_return_the_low_end() {
        let mut rng = Rng::new(9);
        assert_eq!(rng.range_f32(2.0, 2.0), 2.0);
        assert_eq!(rng.range_f64(1.0, f64::INFINITY), 1.0);
    }
}
