//! Colour differences in CIELAB: ΔE\*76, ΔE\*94 and CIEDE2000 `[COL-03]` `[AC-WP02-1]`.
//!
//! All three take two CIELAB triples `[L*, a*, b*]` (see [`super::lab`]) and return a non-negative
//! difference in which 1 is about one just-noticeable step.
//!
//! - **ΔE\*76** (CIE 1976) is the Euclidean distance in CIELAB.
//! - **ΔE\*94** (CIE 116-1995) weights chroma and hue differences by the chroma of the first colour,
//!   the *reference*; it is therefore not symmetric.
//! - **CIEDE2000** (CIE 142-2001) is implemented as formulated in G. Sharma, W. Wu and E. N. Dalal,
//!   "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and
//!   Mathematical Observations", Color Research & Application 30(1), 2005, including their rules for
//!   the hue difference and the mean hue, and is tested against all 34 pairs of that paper's Table 1
//!   to four decimals. It is symmetric.
//!
//! Out-of-range input: any finite Lab triples are accepted. A parametric factor (`kL`, `kC`, `kH`)
//! that is not finite and positive is treated as 1.

use super::lab::{hue_degrees, rgb_to_lab};

/// ΔE\*76: the Euclidean distance between two CIELAB colours.
#[inline]
pub fn delta_e_76(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    (lab1[0] - lab2[0]).hypot(lab1[1] - lab2[1]).hypot(lab1[2] - lab2[2])
}

/// The application constants of ΔE\*94.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct De94Weights {
    /// The lightness factor `kL`.
    pub kl: f64,
    /// The chroma weighting constant `K1` in `SC = 1 + K1·C1`.
    pub k1: f64,
    /// The hue weighting constant `K2` in `SH = 1 + K2·C1`.
    pub k2: f64,
}

impl De94Weights {
    /// The graphic-arts constants of CIE 116-1995: `kL = 1`, `K1 = 0.045`, `K2 = 0.015`.
    pub const GRAPHIC_ARTS: Self = Self { kl: 1.0, k1: 0.045, k2: 0.015 };
    /// The textile constants: `kL = 2`, `K1 = 0.048`, `K2 = 0.014`.
    pub const TEXTILES: Self = Self { kl: 2.0, k1: 0.048, k2: 0.014 };
}

/// A parametric factor as used: itself, or 1 when it is not finite and positive.
#[inline]
fn usable_factor(k: f64) -> f64 {
    if k.is_finite() && k > 0.0 { k } else { 1.0 }
}

/// ΔE\*94 of `sample` against `reference` (the first argument sets the chroma weights):
///
/// `ΔE = √((ΔL/kL)² + (ΔC/SC)² + (ΔH/SH)²)` with `SC = 1 + K1·C1`, `SH = 1 + K2·C1`,
/// `ΔC = C1 − C2` and `ΔH² = Δa² + Δb² − ΔC²`.
///
/// `K1` and `K2` that are not finite and non-negative are treated as 0.
pub fn delta_e_94(reference: [f64; 3], sample: [f64; 3], weights: De94Weights) -> f64 {
    let kl = usable_factor(weights.kl);
    let k1 = if weights.k1.is_finite() && weights.k1 >= 0.0 { weights.k1 } else { 0.0 };
    let k2 = if weights.k2.is_finite() && weights.k2 >= 0.0 { weights.k2 } else { 0.0 };
    let dl = reference[0] - sample[0];
    let da = reference[1] - sample[1];
    let db = reference[2] - sample[2];
    let c1 = reference[1].hypot(reference[2]);
    let c2 = sample[1].hypot(sample[2]);
    let dc = c1 - c2;
    // ΔH² is non-negative in exact arithmetic; rounding can make it slightly negative.
    let dh = (da.hypot(db).powi(2) - dc * dc).max(0.0).sqrt();
    let sc = 1.0 + k1 * c1;
    let sh = 1.0 + k2 * c1;
    (dl / kl).hypot(dc / sc).hypot(dh / sh)
}

/// ΔE\*94 with the graphic-arts constants.
#[inline]
pub fn delta_e_94_graphic_arts(reference: [f64; 3], sample: [f64; 3]) -> f64 {
    delta_e_94(reference, sample, De94Weights::GRAPHIC_ARTS)
}

/// `√(c⁷ / (c⁷ + 25⁷))`, written so that neither a huge nor a tiny chroma overflows.
#[inline]
fn chroma_compensation(c: f64) -> f64 {
    if c > 0.0 { (1.0 / (1.0 + (25.0 / c).powi(7))).sqrt() } else { 0.0 }
}

/// CIEDE2000 with all parametric factors at 1 (the reference conditions).
#[inline]
pub fn delta_e_2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    delta_e_2000_weighted(lab1, lab2, 1.0, 1.0, 1.0)
}

/// CIEDE2000 with the parametric factors `kL`, `kC`, `kH`.
///
/// The steps, in the notation of Sharma, Wu & Dalal (2005), equations 2–22:
///
/// 1. `C̄ = (C1 + C2)/2`, `G = ½(1 − √(C̄⁷/(C̄⁷ + 25⁷)))`, `a′ = (1 + G)·a`, `C′ = √(a′² + b²)`,
///    `h′ = atan2(b, a′)` in `[0°, 360°)`, 0 when `a′ = b = 0`.
/// 2. `ΔL′ = L2 − L1`, `ΔC′ = C′2 − C′1`, `Δh′ = h′2 − h′1` brought into `(−180°, 180°]` (0 when
///    either chroma is 0), `ΔH′ = 2√(C′1·C′2)·sin(Δh′/2)`.
/// 3. `L̄′`, `C̄′` are the means; the mean hue `h̄′` is `(h′1 + h′2)/2`, moved by 180° towards the
///    side of the shorter arc when the hues are more than 180° apart, and `h′1 + h′2` when either
///    chroma is 0.
/// 4. `T = 1 − 0.17cos(h̄′ − 30°) + 0.24cos(2h̄′) + 0.32cos(3h̄′ + 6°) − 0.20cos(4h̄′ − 63°)`,
///    `Δθ = 30·exp(−((h̄′ − 275°)/25)²)`, `RC = 2√(C̄′⁷/(C̄′⁷ + 25⁷))`,
///    `SL = 1 + 0.015(L̄′ − 50)²/√(20 + (L̄′ − 50)²)`, `SC = 1 + 0.045C̄′`, `SH = 1 + 0.015C̄′T`,
///    `RT = −sin(2Δθ)·RC`.
/// 5. `ΔE00 = √((ΔL′/kL·SL)² + (ΔC′/kC·SC)² + (ΔH′/kH·SH)² + RT·(ΔC′/kC·SC)·(ΔH′/kH·SH))`.
pub fn delta_e_2000_weighted(lab1: [f64; 3], lab2: [f64; 3], kl: f64, kc: f64, kh: f64) -> f64 {
    let (kl, kc, kh) = (usable_factor(kl), usable_factor(kc), usable_factor(kh));
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    // Step 1.
    let c_bar = f64::midpoint(a1.hypot(b1), a2.hypot(b2));
    let g = 0.5 * (1.0 - chroma_compensation(c_bar));
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);
    let h1p = hue_degrees(a1p, b1);
    let h2p = hue_degrees(a2p, b2);

    // Step 2.
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let achromatic = c1p * c2p == 0.0;
    let dh = if achromatic {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dh_big = 2.0 * (c1p * c2p).sqrt() * (0.5 * dh).to_radians().sin();

    // Step 3.
    let l_bar = f64::midpoint(l1, l2);
    let c_bar_p = f64::midpoint(c1p, c2p);
    let h_sum = h1p + h2p;
    let h_bar = if achromatic {
        h_sum
    } else if (h1p - h2p).abs() <= 180.0 {
        0.5 * h_sum
    } else if h_sum < 360.0 {
        0.5 * h_sum + 180.0
    } else {
        0.5 * h_sum - 180.0
    };

    // Step 4.
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let d_theta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * chroma_compensation(c_bar_p);
    let l50 = (l_bar - 50.0) * (l_bar - 50.0);
    let sl = 1.0 + 0.015 * l50 / (20.0 + l50).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -(2.0 * d_theta).to_radians().sin() * rc;

    // Step 5. The quadratic form is positive semi-definite because |RT| ≤ 2; the clamp only removes
    // a rounding residue below zero.
    let tl = dl / (kl * sl);
    let tc = dc / (kc * sc);
    let th = dh_big / (kh * sh);
    (tl * tl + tc * tc + th * th + rt * tc * th).max(0.0).sqrt()
}

/// CIEDE2000 between two linear Rec.709 colours (through CIELAB under D65).
#[inline]
pub fn delta_e_2000_rgb(rgb1: [f64; 3], rgb2: [f64; 3]) -> f64 {
    delta_e_2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))
}

/// ΔE\*76 between two linear Rec.709 colours (through CIELAB under D65).
#[inline]
pub fn delta_e_76_rgb(rgb1: [f64; 3], rgb2: [f64; 3]) -> f64 {
    delta_e_76(rgb_to_lab(rgb1), rgb_to_lab(rgb2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    /// Table 1 of Sharma, Wu & Dalal (2005), also distributed by the authors as
    /// `ciede2000testdata.txt`: `L1 a1 b1 L2 a2 b2 ΔE00`, all 34 pairs.
    const SHARMA_PAIRS: [[f64; 7]; 34] = [
        [50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425],
        [50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615],
        [50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412],
        [50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669],
        [50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461],
        [50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065],
        [50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
        [50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977],
        [50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030],
        [50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000],
        [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
        [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630],
        [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731],
        [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
        [22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373],
        [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
        [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
        [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
        [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
        [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
    ];

    #[test]
    fn ac_wp02_1_ciede2000_matches_all_34_sharma_pairs_to_four_decimals() {
        for (index, row) in SHARMA_PAIRS.iter().enumerate() {
            let lab1 = [row[0], row[1], row[2]];
            let lab2 = [row[3], row[4], row[5]];
            let published = row[6];
            for got in [delta_e_2000(lab1, lab2), delta_e_2000(lab2, lab1)] {
                let rounded = (got * 1e4).round() / 1e4;
                assert!(
                    (rounded - published).abs() < 1e-9,
                    "pair {}: computed {got:.6}, published {published:.4}",
                    index + 1
                );
            }
        }
    }

    #[test]
    fn ciede2000_is_a_symmetric_non_negative_difference() {
        let mut rng = Rng::new(0xDE00);
        for _ in 0..20_000 {
            let lab1 = [rng.range_f64(0.0, 100.0), rng.range_f64(-120.0, 120.0), rng.range_f64(-120.0, 120.0)];
            let lab2 = [rng.range_f64(0.0, 100.0), rng.range_f64(-120.0, 120.0), rng.range_f64(-120.0, 120.0)];
            let forward = delta_e_2000(lab1, lab2);
            let backward = delta_e_2000(lab2, lab1);
            assert!(forward >= 0.0 && forward.is_finite());
            assert!((forward - backward).abs() < 1e-9, "{lab1:?} {lab2:?}");
            assert_eq!(delta_e_2000(lab1, lab1), 0.0);
            // For close colours it is of the order of ΔE*76 (never more than about that).
            assert!(forward <= delta_e_76(lab1, lab2) * 1.5 + 1e-9);
        }
    }

    #[test]
    fn ciede2000_parametric_factors() {
        let lab1 = [50.0, 2.5, 0.0];
        let lab2 = [61.0, -5.0, 29.0];
        let base = delta_e_2000(lab1, lab2);
        assert_eq!(delta_e_2000_weighted(lab1, lab2, 1.0, 1.0, 1.0), base);
        // A pure lightness difference scales with 1/kL: at L̄ = 50, SL = 1.
        let light = delta_e_2000_weighted([45.0, 0.0, 0.0], [55.0, 0.0, 0.0], 2.0, 1.0, 1.0);
        assert!((light - 5.0).abs() < 1e-12);
        assert!((delta_e_2000([45.0, 0.0, 0.0], [55.0, 0.0, 0.0]) - 10.0).abs() < 1e-12);
        // Larger factors only shrink the difference.
        assert!(delta_e_2000_weighted(lab1, lab2, 2.0, 1.0, 1.0) < base);
        assert!(delta_e_2000_weighted(lab1, lab2, 1.0, 2.0, 1.0) < base);
        assert!(delta_e_2000_weighted(lab1, lab2, 1.0, 1.0, 2.0) < base);
        // Unusable factors fall back to 1.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(delta_e_2000_weighted(lab1, lab2, bad, bad, bad), base);
        }
    }

    #[test]
    fn delta_e_76_is_the_euclidean_distance() {
        assert_eq!(delta_e_76([50.0, 0.0, 0.0], [50.0, 0.0, 0.0]), 0.0);
        assert!((delta_e_76([50.0, 10.0, -5.0], [53.0, 14.0, 7.0]) - 13.0).abs() < 1e-12);
        assert!((delta_e_76([0.0, 0.0, 0.0], [100.0, 0.0, 0.0]) - 100.0).abs() < 1e-12);
        // Black to white through the RGB helper.
        assert!((delta_e_76_rgb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) - 100.0).abs() < 1e-9);
        assert!((delta_e_2000_rgb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) - 100.0).abs() < 1e-9);
        assert_eq!(delta_e_2000_rgb([0.2, 0.3, 0.4], [0.2, 0.3, 0.4]), 0.0);
    }

    #[test]
    fn delta_e_94_worked_cases() {
        // Lightness only: ΔL/kL.
        assert!((delta_e_94_graphic_arts([50.0, 0.0, 0.0], [53.0, 0.0, 0.0]) - 3.0).abs() < 1e-12);
        assert!((delta_e_94([50.0, 0.0, 0.0], [53.0, 0.0, 0.0], De94Weights::TEXTILES) - 1.5).abs() < 1e-12);
        // Chroma only: C1 = 10, C2 = 20, SC = 1 + 0.045·10 = 1.45, ΔE = 10/1.45.
        let chroma = delta_e_94_graphic_arts([50.0, 10.0, 0.0], [50.0, 20.0, 0.0]);
        assert!((chroma - 10.0 / 1.45).abs() < 1e-12);
        // The reference sets the weights: swapped, SC = 1 + 0.045·20 = 1.9.
        let swapped = delta_e_94_graphic_arts([50.0, 20.0, 0.0], [50.0, 10.0, 0.0]);
        assert!((swapped - 10.0 / 1.9).abs() < 1e-12);
        // Hue only: equal chroma 10, ΔH = √200, SH = 1 + 0.015·10 = 1.15.
        let hue = delta_e_94_graphic_arts([50.0, 10.0, 0.0], [50.0, 0.0, 10.0]);
        assert!((hue - 200.0_f64.sqrt() / 1.15).abs() < 1e-12);
        // Textiles: SH = 1 + 0.014·10.
        let hue = delta_e_94([50.0, 10.0, 0.0], [50.0, 0.0, 10.0], De94Weights::TEXTILES);
        assert!((hue - 200.0_f64.sqrt() / 1.14).abs() < 1e-12);
        // For neutral references it reduces to ΔE*76.
        let mut rng = Rng::new(0xDE94);
        for _ in 0..2000 {
            let l = rng.range_f64(0.0, 100.0);
            let sample = [rng.range_f64(0.0, 100.0), rng.range_f64(-1.0, 1.0), 0.0];
            let reference = [l, 0.0, 0.0];
            let d94 = delta_e_94_graphic_arts(reference, sample);
            assert!((d94 - delta_e_76(reference, sample)).abs() < 1e-9);
            assert!(d94 <= delta_e_76(reference, sample) + 1e-12);
        }
        // Unusable weights fall back (kL to 1, K1 and K2 to 0, which is ΔE*76).
        let bad = De94Weights { kl: f64::NAN, k1: -1.0, k2: f64::INFINITY };
        let (a, b) = ([40.0, 20.0, -10.0], [45.0, -3.0, 12.0]);
        assert!((delta_e_94(a, b, bad) - delta_e_76(a, b)).abs() < 1e-12);
    }

    #[test]
    fn extreme_finite_input_stays_finite() {
        for v in [1e100, -1e100, 1e-300, -1e-300] {
            for (a, b) in [([v, v, v], [0.0, 0.0, 0.0]), ([v, -v, v], [-v, v, -v]), ([50.0, v, 0.0], [50.0, 0.0, v])] {
                assert!(delta_e_76(a, b).is_finite(), "{v}");
                assert!(delta_e_94_graphic_arts(a, b).is_finite(), "{v}");
                assert!(delta_e_2000(a, b).is_finite(), "{v}");
                assert!(delta_e_2000(b, a).is_finite(), "{v}");
            }
        }
        // NaN input never panics.
        let _ = delta_e_2000([f64::NAN, 0.0, 0.0], [50.0, f64::INFINITY, 0.0]);
        let _ = delta_e_94_graphic_arts([f64::NAN, 0.0, 0.0], [50.0, f64::INFINITY, 0.0]);
    }
}
