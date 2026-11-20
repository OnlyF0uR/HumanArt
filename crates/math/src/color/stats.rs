//! Streaming colour statistics: per-region mean, variance and range in linear RGB and CIELAB, colour
//! difference summaries and fixed-bin histograms `[COL-03]`.
//!
//! The quality metrics of section 10.8 report per-region colour statistics in CIELAB and CIEDE2000
//! against a reference; the seam-visibility metric summarises ΔE over seam pixel pairs; the scopes
//! need histograms. Everything here is a fixed-size accumulator: pushing a sample never allocates,
//! partial results from tiles or threads combine with `merge`, and the order of samples changes the
//! result only by rounding.
//!
//! Means and variances use the weighted form of Welford's update (D. H. D. West, "Updating mean and
//! variance estimates: an improved method", Communications of the ACM 22(9), 1979) and the pairwise
//! combination of T. F. Chan, G. H. Golub and R. J. LeVeque (1979) for `merge`, which stay accurate
//! where the textbook `Σx² − (Σx)²/n` cancels. Variances are population variances (divided by the
//! total weight), the statistic of a region's pixels rather than an estimate from a sample.
//!
//! Samples with a non-finite component, and weights that are not finite and positive, are skipped
//! and counted in `rejected`.

use super::delta_e::delta_e_2000;
use super::lab::{lab_to_lch, rgb_to_lab};

/// Weighted running mean, second central moment and range of a three-component quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moments3 {
    /// The total weight of the accepted samples.
    pub weight: f64,
    /// The weighted mean.
    pub mean: [f64; 3],
    /// The weighted sum of squared deviations from the mean (`M2`).
    pub m2: [f64; 3],
    /// The component-wise minimum (`+∞` while empty).
    pub min: [f64; 3],
    /// The component-wise maximum (`−∞` while empty).
    pub max: [f64; 3],
}

impl Moments3 {
    /// No samples.
    pub const EMPTY: Self =
        Self { weight: 0.0, mean: [0.0; 3], m2: [0.0; 3], min: [f64::INFINITY; 3], max: [f64::NEG_INFINITY; 3] };

    /// Adds a sample with weight `w`; the caller has checked that both are finite and `w > 0`.
    fn push(&mut self, x: [f64; 3], w: f64) {
        self.weight += w;
        let f = w / self.weight;
        for (c, &xc) in x.iter().enumerate() {
            let delta = xc - self.mean[c];
            self.mean[c] += f * delta;
            self.m2[c] += w * delta * (xc - self.mean[c]);
            self.min[c] = self.min[c].min(xc);
            self.max[c] = self.max[c].max(xc);
        }
    }

    /// Combines two accumulators as if all samples had been pushed into one.
    pub fn merge(&mut self, other: &Self) {
        if other.weight <= 0.0 {
            return;
        }
        if self.weight <= 0.0 {
            *self = *other;
            return;
        }
        let total = self.weight + other.weight;
        let fb = other.weight / total;
        for c in 0..3 {
            let delta = other.mean[c] - self.mean[c];
            self.mean[c] += delta * fb;
            self.m2[c] += other.m2[c] + delta * delta * self.weight * fb;
            self.min[c] = self.min[c].min(other.min[c]);
            self.max[c] = self.max[c].max(other.max[c]);
        }
        self.weight = total;
    }

    /// The mean, or `None` without samples.
    pub fn mean(&self) -> Option<[f64; 3]> {
        (self.weight > 0.0).then_some(self.mean)
    }

    /// The population variance per component, or `None` without samples.
    pub fn variance(&self) -> Option<[f64; 3]> {
        (self.weight > 0.0).then(|| self.m2.map(|m| (m / self.weight).max(0.0)))
    }

    /// The population standard deviation per component, or `None` without samples.
    pub fn std_dev(&self) -> Option<[f64; 3]> {
        self.variance().map(|v| v.map(f64::sqrt))
    }

    /// The component-wise `(min, max)`, or `None` without samples.
    pub fn range(&self) -> Option<([f64; 3], [f64; 3])> {
        (self.weight > 0.0).then_some((self.min, self.max))
    }
}

impl Default for Moments3 {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Whether a sample and its weight may be accumulated.
#[inline]
fn usable(x: [f64; 3], w: f64) -> bool {
    w.is_finite() && w > 0.0 && x.iter().all(|v| v.is_finite())
}

/// The colour statistics of a region: moments in linear Rec.709 RGB and in CIELAB (D65).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorStats {
    /// The moments in linear Rec.709 RGB.
    pub rgb: Moments3,
    /// The moments in CIELAB `[L*, a*, b*]` under D65.
    pub lab: Moments3,
    /// The number of accepted samples.
    pub count: u64,
    /// The number of samples skipped for a non-finite component or an unusable weight.
    pub rejected: u64,
}

impl ColorStats {
    /// No samples.
    pub const EMPTY: Self = Self { rgb: Moments3::EMPTY, lab: Moments3::EMPTY, count: 0, rejected: 0 };

    /// Adds a linear Rec.709 colour with weight 1.
    #[inline]
    pub fn push(&mut self, rgb: [f64; 3]) {
        self.push_weighted(rgb, 1.0);
    }

    /// Adds a linear Rec.709 colour with weight `w` (a soft region mask, for example).
    pub fn push_weighted(&mut self, rgb: [f64; 3], w: f64) {
        if !usable(rgb, w) {
            self.rejected += 1;
            return;
        }
        let lab = rgb_to_lab(rgb);
        if !lab.iter().all(|v| v.is_finite()) {
            self.rejected += 1;
            return;
        }
        self.rgb.push(rgb, w);
        self.lab.push(lab, w);
        self.count += 1;
    }

    /// Adds an `f32` linear Rec.709 colour with weight 1.
    #[inline]
    pub fn push_f32(&mut self, rgb: [f32; 3]) {
        self.push_weighted(super::matrix::widen(rgb), 1.0);
    }

    /// Combines two accumulators.
    pub fn merge(&mut self, other: &Self) {
        self.rgb.merge(&other.rgb);
        self.lab.merge(&other.lab);
        self.count += other.count;
        self.rejected += other.rejected;
    }

    /// The mean CIELAB of the samples (not the CIELAB of the mean RGB), or `None` without samples.
    pub fn mean_lab(&self) -> Option<[f64; 3]> {
        self.lab.mean()
    }

    /// The mean CIELAB as LCh `[L*, C*, h°]`, or `None` without samples. The hue is that of the
    /// mean `(a*, b*)`, so opposite hues cancel instead of averaging their angles.
    pub fn mean_lch(&self) -> Option<[f64; 3]> {
        self.lab.mean().map(lab_to_lch)
    }

    /// The mean linear RGB, or `None` without samples.
    pub fn mean_rgb(&self) -> Option<[f64; 3]> {
        self.rgb.mean()
    }

    /// The CIELAB standard deviation `[σL*, σa*, σb*]`, or `None` without samples.
    pub fn std_dev_lab(&self) -> Option<[f64; 3]> {
        self.lab.std_dev()
    }

    /// The CIEDE2000 between this region's mean CIELAB and a reference CIELAB, or `None` without
    /// samples.
    pub fn delta_e_2000_to(&self, reference_lab: [f64; 3]) -> Option<f64> {
        self.mean_lab().map(|lab| delta_e_2000(lab, reference_lab))
    }
}

/// A streaming summary of colour differences: count, mean, root mean square, maximum and the count
/// above a threshold (the seam-visibility metric, CIEDE2000 against a reference image).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaEStats {
    /// The threshold for [`Self::above_threshold`].
    pub threshold: f64,
    /// The number of accepted differences.
    pub count: u64,
    /// The number of differences above `threshold`.
    pub above_threshold: u64,
    /// The number of skipped pairs (non-finite input).
    pub rejected: u64,
    sum: f64,
    sum_sq: f64,
    max: f64,
}

impl DeltaEStats {
    /// An empty summary counting differences above `threshold` (a non-finite threshold counts none).
    pub const fn new(threshold: f64) -> Self {
        Self { threshold, count: 0, above_threshold: 0, rejected: 0, sum: 0.0, sum_sq: 0.0, max: 0.0 }
    }

    /// Adds one difference value; negative or non-finite values are rejected.
    pub fn push_delta_e(&mut self, delta_e: f64) {
        if !(delta_e.is_finite() && delta_e >= 0.0) {
            self.rejected += 1;
            return;
        }
        self.count += 1;
        self.sum += delta_e;
        self.sum_sq += delta_e * delta_e;
        self.max = self.max.max(delta_e);
        if delta_e > self.threshold {
            self.above_threshold += 1;
        }
    }

    /// Adds the CIEDE2000 between two CIELAB colours.
    pub fn push_lab(&mut self, a: [f64; 3], b: [f64; 3]) {
        if usable(a, 1.0) && usable(b, 1.0) {
            self.push_delta_e(delta_e_2000(a, b));
        } else {
            self.rejected += 1;
        }
    }

    /// Adds the CIEDE2000 between two linear Rec.709 colours.
    pub fn push_rgb(&mut self, a: [f64; 3], b: [f64; 3]) {
        if usable(a, 1.0) && usable(b, 1.0) {
            self.push_delta_e(delta_e_2000(rgb_to_lab(a), rgb_to_lab(b)));
        } else {
            self.rejected += 1;
        }
    }

    /// Combines two summaries; the threshold of `self` is kept, so both should share it.
    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.above_threshold += other.above_threshold;
        self.rejected += other.rejected;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.max = self.max.max(other.max);
    }

    /// The mean difference, or `None` without samples.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }

    /// The root mean square difference, or `None` without samples.
    pub fn rms(&self) -> Option<f64> {
        (self.count > 0).then(|| (self.sum_sq / self.count as f64).sqrt())
    }

    /// The largest difference, or `None` without samples.
    pub fn max(&self) -> Option<f64> {
        (self.count > 0).then_some(self.max)
    }

    /// The fraction of differences above the threshold, or `None` without samples.
    pub fn fraction_above_threshold(&self) -> Option<f64> {
        (self.count > 0).then(|| self.above_threshold as f64 / self.count as f64)
    }
}

/// A histogram of `N` equal bins over `[lo, hi)`, with the values below and above counted apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Histogram<const N: usize> {
    lo: f64,
    hi: f64,
    /// The bin counts.
    pub bins: [u64; N],
    /// Values below `lo`.
    pub below: u64,
    /// Values at or above `hi`.
    pub above: u64,
    /// NaN values.
    pub rejected: u64,
}

impl<const N: usize> Histogram<N> {
    /// An empty histogram over `[lo, hi)`. `None` when the bounds are not finite, `lo ≥ hi` or
    /// `N = 0`.
    pub fn new(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo < hi && N > 0).then_some(Self {
            lo,
            hi,
            bins: [0; N],
            below: 0,
            above: 0,
            rejected: 0,
        })
    }

    /// The range `[lo, hi)`.
    pub const fn range(&self) -> [f64; 2] {
        [self.lo, self.hi]
    }

    /// The bin of `value`, or `None` when it falls outside the range (or is NaN).
    pub fn bin_of(&self, value: f64) -> Option<usize> {
        if !(value >= self.lo && value < self.hi) {
            return None;
        }
        let t = (value - self.lo) / (self.hi - self.lo);
        // `t < 1` here, but `t · N` can round up to `N` for the last representable values.
        Some(((t * N as f64) as usize).min(N - 1))
    }

    /// Counts one value.
    pub fn push(&mut self, value: f64) {
        if value.is_nan() {
            self.rejected += 1;
        } else if let Some(bin) = self.bin_of(value) {
            self.bins[bin] += 1;
        } else if value < self.lo {
            self.below += 1;
        } else {
            self.above += 1;
        }
    }

    /// Adds the counts of another histogram with the same range and bins; returns `false` and changes
    /// nothing when the ranges differ.
    pub fn merge(&mut self, other: &Self) -> bool {
        if self.lo != other.lo || self.hi != other.hi {
            return false;
        }
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            *a += b;
        }
        self.below += other.below;
        self.above += other.above;
        self.rejected += other.rejected;
        true
    }

    /// The number of counted (non-NaN) values, out-of-range ones included.
    pub fn total(&self) -> u64 {
        self.bins.iter().sum::<u64>() + self.below + self.above
    }

    /// The value below which the fraction `p` (in `[0, 1]`, clamped) of the counted values lie,
    /// interpolated linearly inside a bin; values below the range count at `lo`, above it at `hi`.
    /// `None` when nothing was counted or `p` is NaN.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let total = self.total();
        if total == 0 || p.is_nan() {
            return None;
        }
        let target = p.clamp(0.0, 1.0) * total as f64;
        let mut seen = self.below as f64;
        if target <= seen && self.below > 0 {
            return Some(self.lo);
        }
        let width = (self.hi - self.lo) / N as f64;
        for (i, &count) in self.bins.iter().enumerate() {
            let next = seen + count as f64;
            if count > 0 && target <= next {
                let within = ((target - seen) / count as f64).clamp(0.0, 1.0);
                return Some(self.lo + (i as f64 + within) * width);
            }
            seen = next;
        }
        Some(self.hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn moments_match_a_two_pass_computation() {
        let mut rng = Rng::new(0x57A7);
        let samples: Vec<[f64; 3]> =
            (0..5000).map(|_| [rng.range_f64(0.0, 1.0), rng.range_f64(0.0, 0.5), rng.range_f64(0.2, 0.3)]).collect();
        let mut stats = ColorStats::EMPTY;
        for s in &samples {
            stats.push(*s);
        }
        assert_eq!(stats.count, 5000);
        let labs: Vec<[f64; 3]> = samples.iter().map(|s| rgb_to_lab(*s)).collect();
        let n = labs.len() as f64;
        let mean: [f64; 3] = std::array::from_fn(|c| labs.iter().map(|l| l[c]).sum::<f64>() / n);
        let var: [f64; 3] = std::array::from_fn(|c| labs.iter().map(|l| (l[c] - mean[c]).powi(2)).sum::<f64>() / n);
        let got_mean = stats.mean_lab().unwrap();
        let got_var = stats.lab.variance().unwrap();
        for c in 0..3 {
            assert!((got_mean[c] - mean[c]).abs() < 1e-9);
            assert!((got_var[c] - var[c]).abs() < 1e-7 * var[c].max(1.0));
            let min = labs.iter().map(|l| l[c]).fold(f64::INFINITY, f64::min);
            assert_eq!(stats.lab.range().unwrap().0[c], min);
        }
        let rgb_mean = stats.mean_rgb().unwrap();
        assert!((rgb_mean[2] - 0.25).abs() < 0.01);
    }

    #[test]
    fn weights_and_merge_equal_a_single_pass() {
        let mut rng = Rng::new(0x3E6E);
        let mut whole = ColorStats::EMPTY;
        let mut parts = [ColorStats::EMPTY; 4];
        for i in 0..4000 {
            let rgb = [rng.range_f64(0.0, 2.0), rng.range_f64(0.0, 1.0), rng.range_f64(0.0, 1.0)];
            let w = rng.range_f64(0.1, 3.0);
            whole.push_weighted(rgb, w);
            parts[i % 4].push_weighted(rgb, w);
        }
        let mut merged = ColorStats::EMPTY;
        for p in &parts {
            merged.merge(p);
        }
        merged.merge(&ColorStats::EMPTY);
        assert_eq!(merged.count, whole.count);
        for c in 0..3 {
            assert!((merged.lab.mean[c] - whole.lab.mean[c]).abs() < 1e-9);
            assert!((merged.lab.m2[c] - whole.lab.m2[c]).abs() < 1e-6 * whole.lab.m2[c]);
            assert!((merged.rgb.weight - whole.rgb.weight).abs() < 1e-9);
        }
        // Weight 2 is the same as pushing twice.
        let mut twice = ColorStats::EMPTY;
        let mut weighted = ColorStats::EMPTY;
        for rgb in [[0.1, 0.2, 0.3], [0.5, 0.1, 0.0]] {
            twice.push(rgb);
            twice.push(rgb);
            weighted.push_weighted(rgb, 2.0);
        }
        for c in 0..3 {
            assert!((twice.lab.mean[c] - weighted.lab.mean[c]).abs() < 1e-12);
            assert!((twice.lab.variance().unwrap()[c] - weighted.lab.variance().unwrap()[c]).abs() < 1e-9);
        }
    }

    #[test]
    fn constant_regions_have_zero_spread_and_lch_of_the_mean() {
        let mut stats = ColorStats::EMPTY;
        for _ in 0..100 {
            stats.push_f32([0.4, 0.2, 0.1]);
        }
        assert!(stats.std_dev_lab().unwrap().iter().all(|s| *s < 1e-6));
        let lab = rgb_to_lab(super::super::matrix::widen([0.4_f32, 0.2, 0.1]));
        assert!(stats.delta_e_2000_to(lab).unwrap() < 1e-9);
        assert_eq!(stats.mean_lch().unwrap(), lab_to_lch(stats.mean_lab().unwrap()));
        // Empty stats give no numbers.
        let empty = ColorStats::EMPTY;
        assert_eq!(empty.mean_lab(), None);
        assert_eq!(empty.std_dev_lab(), None);
        assert_eq!(empty.lab.range(), None);
        assert_eq!(empty.delta_e_2000_to([50.0, 0.0, 0.0]), None);
    }

    #[test]
    fn unusable_samples_are_rejected() {
        let mut stats = ColorStats::EMPTY;
        stats.push([f64::NAN, 0.0, 0.0]);
        stats.push([f64::INFINITY, 0.0, 0.0]);
        stats.push_weighted([0.1; 3], 0.0);
        stats.push_weighted([0.1; 3], -1.0);
        stats.push_weighted([0.1; 3], f64::NAN);
        stats.push([0.2; 3]);
        assert_eq!((stats.count, stats.rejected), (1, 5));
        let mut d = DeltaEStats::new(1.0);
        d.push_delta_e(f64::NAN);
        d.push_delta_e(-1.0);
        d.push_lab([f64::NAN, 0.0, 0.0], [50.0, 0.0, 0.0]);
        d.push_rgb([0.1; 3], [f64::INFINITY, 0.0, 0.0]);
        assert_eq!((d.count, d.rejected), (0, 4));
        assert_eq!(d.mean(), None);
        assert_eq!(d.max(), None);
    }

    #[test]
    fn delta_e_summary_values() {
        let mut d = DeltaEStats::new(2.0);
        for v in [1.0, 3.0, 2.0, 4.0] {
            d.push_delta_e(v);
        }
        assert_eq!(d.mean(), Some(2.5));
        assert_eq!(d.max(), Some(4.0));
        assert!((d.rms().unwrap() - 7.5_f64.sqrt()).abs() < 1e-15);
        assert_eq!(d.fraction_above_threshold(), Some(0.5));
        let mut other = DeltaEStats::new(2.0);
        other.push_lab([50.0, 0.0, 0.0], [60.0, 0.0, 0.0]);
        other.push_rgb([0.2; 3], [0.2; 3]);
        d.merge(&other);
        assert_eq!(d.count, 6);
        assert!((d.max().unwrap() - delta_e_2000([50.0, 0.0, 0.0], [60.0, 0.0, 0.0])).abs() < 1e-12);
        assert_eq!(d.above_threshold, 3);
    }

    #[test]
    fn histogram_bins_percentiles_and_merge() {
        assert!(Histogram::<4>::new(1.0, 1.0).is_none());
        assert!(Histogram::<4>::new(0.0, f64::INFINITY).is_none());
        assert!(Histogram::<0>::new(0.0, 1.0).is_none());
        let mut h = Histogram::<4>::new(0.0, 1.0).unwrap();
        for v in [0.0, 0.1, 0.3, 0.6, 0.99, 1.0, -0.5, f64::NAN] {
            h.push(v);
        }
        assert_eq!(h.bins, [2, 1, 1, 1]);
        assert_eq!((h.below, h.above, h.rejected), (1, 1, 1));
        assert_eq!(h.total(), 7);
        assert_eq!(h.bin_of(1.0 - f64::EPSILON / 2.0), Some(3));
        assert_eq!(h.percentile(0.0), Some(0.0));
        assert_eq!(h.percentile(1.0), Some(1.0));
        assert_eq!(h.percentile(f64::NAN), None);
        // Uniform data: the median is the middle of the range.
        let mut u = Histogram::<100>::new(0.0, 10.0).unwrap();
        for i in 0..10_000 {
            u.push(f64::from(i) * 1e-3);
        }
        assert!((u.percentile(0.5).unwrap() - 5.0).abs() < 1e-9);
        assert!((u.percentile(0.9).unwrap() - 9.0).abs() < 1e-9);
        let copy = u;
        assert!(u.merge(&copy));
        assert_eq!(u.total(), 20_000);
        assert!(!u.merge(&Histogram::<100>::new(0.0, 5.0).unwrap()));
        assert_eq!(Histogram::<3>::new(0.0, 1.0).unwrap().percentile(0.5), None);
        assert_eq!(h.range(), [0.0, 1.0]);
    }
}
