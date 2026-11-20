//! Image statistics for the micro-detail tests `[AC-WP13-2]`: an in-house radix-2 FFT, the
//! radially and angularly binned power spectrum, the circular autocorrelation and the tiling
//! detector.
//!
//! The **tiling detector** looks for repetition the way an eye does: the normalised
//! autocorrelation `ρ(lag) = Σ (I(p) - μ)(I(p + lag) - μ) / Σ (I - μ)²` of the synthesised surface
//! at the lags of the exemplar period (and its diagonal). Plain repetition gives `ρ ≈ 1` there; a
//! texture with no visible repeat gives values near those of unrelated content (`|ρ|` well below
//! 0.2). The score is the largest `|ρ|` over those lags.

use super::error::SkinError;

/// Largest FFT side.
pub const MAX_FFT_SIDE: usize = 4096;

/// A complex number.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    fn mul(self, o: Self) -> Self {
        Self { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }

    /// `|z|²`.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// In-place iterative radix-2 FFT of `data` (length a power of two). `inverse` computes the
/// unnormalised inverse transform.
pub fn fft(data: &mut [Complex], inverse: bool) -> Result<(), SkinError> {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() || n > MAX_FFT_SIDE {
        return Err(SkinError::Invalid { what: "FFT length must be a power of two up to 4096" });
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) };
        if j > i {
            data.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * std::f64::consts::TAU / len as f64;
        let step = Complex { re: angle.cos(), im: angle.sin() };
        for start in (0..n).step_by(len) {
            let mut w = Complex { re: 1.0, im: 0.0 };
            for k in 0..len / 2 {
                let a = data[start + k];
                let b = data[start + k + len / 2].mul(w);
                data[start + k] = Complex { re: a.re + b.re, im: a.im + b.im };
                data[start + k + len / 2] = Complex { re: a.re - b.re, im: a.im - b.im };
                w = w.mul(step);
            }
        }
        len *= 2;
    }
    Ok(())
}

/// 2D FFT of a `size × size` row-major image (power-of-two side).
pub fn fft2(data: &mut [Complex], size: usize, inverse: bool) -> Result<(), SkinError> {
    if data.len() != size * size {
        return Err(SkinError::Invalid { what: "2D FFT data is not size x size" });
    }
    for row in data.chunks_exact_mut(size) {
        fft(row, inverse)?;
    }
    let mut column = vec![Complex::default(); size];
    for x in 0..size {
        for (y, c) in column.iter_mut().enumerate() {
            *c = data[y * size + x];
        }
        fft(&mut column, inverse)?;
        for (y, c) in column.iter().enumerate() {
            data[y * size + x] = *c;
        }
    }
    Ok(())
}

/// The power spectrum `|F(k)|²` of a mean-removed `size × size` image, normalised so the total is
/// the image's variance.
pub fn power_spectrum(image: &[f32], size: usize) -> Result<Vec<f64>, SkinError> {
    if image.len() != size * size || size == 0 {
        return Err(SkinError::Invalid { what: "image is not size x size" });
    }
    let mean = image.iter().map(|&v| f64::from(v)).sum::<f64>() / image.len() as f64;
    let mut data: Vec<Complex> = image.iter().map(|&v| Complex { re: f64::from(v) - mean, im: 0.0 }).collect();
    fft2(&mut data, size, false)?;
    let n2 = (size * size) as f64;
    Ok(data.iter().map(|c| c.norm_sqr() / (n2 * n2)).collect())
}

/// The signed frequency index of bin `i` of a length-`size` transform.
fn signed(i: usize, size: usize) -> f64 {
    if i <= size / 2 { i as f64 } else { i as f64 - size as f64 }
}

/// Radially binned power: bin `r` holds the power at radial frequency `r..r+1` cycles per image,
/// for `r` in `0..size / 2`.
pub fn radial_power(spectrum: &[f64], size: usize) -> Vec<f64> {
    let mut bins = vec![0.0; size / 2];
    for y in 0..size {
        for x in 0..size {
            let r = signed(x, size).hypot(signed(y, size));
            let b = r as usize;
            if let (Some(bin), Some(p)) = (bins.get_mut(b), spectrum.get(y * size + x)) {
                *bin += p;
            }
        }
    }
    bins
}

/// Power binned by the orientation of the frequency vector in `bins` bins over 0..180°, counting
/// only radial frequencies in `r_min..r_max` cycles per image.
pub fn angular_power(spectrum: &[f64], size: usize, bins: usize, r_min: f64, r_max: f64) -> Vec<f64> {
    let mut out = vec![0.0; bins.max(1)];
    let n = out.len();
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (signed(x, size), signed(y, size));
            let r = fx.hypot(fy);
            if r < r_min || r >= r_max {
                continue;
            }
            let angle = fy.atan2(fx).rem_euclid(std::f64::consts::PI);
            let b = ((angle / std::f64::consts::PI) * n as f64) as usize % n;
            if let Some(p) = spectrum.get(y * size + x) {
                out[b] += p;
            }
        }
    }
    out
}

/// The circular normalised autocorrelation of a `size × size` image (power-of-two side): entry
/// `y * size + x` is `ρ` at lag `(x, y)` (wrapping), with `ρ(0) = 1`.
pub fn autocorrelation(image: &[f32], size: usize) -> Result<Vec<f64>, SkinError> {
    let spectrum = power_spectrum(image, size)?;
    let mut data: Vec<Complex> = spectrum.iter().map(|&p| Complex { re: p, im: 0.0 }).collect();
    fft2(&mut data, size, true)?;
    let zero = data.first().map_or(0.0, |c| c.re);
    if zero <= 0.0 {
        return Err(SkinError::Invalid { what: "a constant image has no autocorrelation" });
    }
    Ok(data.iter().map(|c| c.re / zero).collect())
}

/// The tiling score of a `size × size` synthesised surface made from an exemplar repeating every
/// `period` texels: the largest `|ρ|` at lags `(period, 0)`, `(0, period)` and `(period, period)`
/// and their multiples below `size / 2`.
pub fn tiling_score(image: &[f32], size: usize, period: usize) -> Result<f64, SkinError> {
    if period == 0 || period >= size {
        return Err(SkinError::Invalid { what: "the tiling period must be in 1..size" });
    }
    let rho = autocorrelation(image, size)?;
    let mut score: f64 = 0.0;
    let mut k = period;
    while k <= size / 2 {
        for (x, y) in [(k, 0), (0, k), (k, k)] {
            if let Some(v) = rho.get(y * size + x) {
                score = score.max(v.abs());
            }
        }
        k += period;
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    #[test]
    fn fft_matches_a_direct_dft_and_inverts() {
        let mut rng = Rng::new(3);
        let n = 16;
        let input: Vec<Complex> = (0..n).map(|_| Complex { re: rng.normal(), im: rng.normal() }).collect();
        let mut fast = input.clone();
        fft(&mut fast, false).unwrap_or_else(|e| panic!("{e}"));
        for (k, f) in fast.iter().enumerate() {
            let mut sum = Complex::default();
            for (j, x) in input.iter().enumerate() {
                let a = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                sum.re += x.re * a.cos() - x.im * a.sin();
                sum.im += x.re * a.sin() + x.im * a.cos();
            }
            assert!((sum.re - f.re).abs() < 1e-9 && (sum.im - f.im).abs() < 1e-9);
        }
        fft(&mut fast, true).unwrap_or_else(|e| panic!("{e}"));
        for (a, b) in fast.iter().zip(&input) {
            assert!((a.re / n as f64 - b.re).abs() < 1e-12);
        }
        assert!(fft(&mut [Complex::default(); 3], false).is_err());
    }

    #[test]
    fn parseval_and_a_pure_wave() {
        let size = 64;
        // A wave with 5 cycles along x: all power in radial bin 5, angle 0.
        let image: Vec<f32> =
            (0..size * size).map(|i| ((i % size) as f32 * std::f32::consts::TAU * 5.0 / size as f32).sin()).collect();
        let spectrum = power_spectrum(&image, size).unwrap_or_else(|e| panic!("{e}"));
        let variance: f64 = image.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (size * size) as f64;
        let total: f64 = spectrum.iter().sum();
        assert!((total - variance).abs() < 1e-6, "Parseval");
        let radial = radial_power(&spectrum, size);
        assert!(radial[5] / total > 0.999);
        let angular = angular_power(&spectrum, size, 8, 1.0, 32.0);
        assert!(angular[0] / total > 0.999);
    }

    #[test]
    fn tiling_detector_separates_repeats_from_noise() {
        let size = 128;
        let mut rng = Rng::new(11);
        let tile: Vec<f32> = (0..32 * 32).map(|_| rng.f32()).collect();
        let repeated: Vec<f32> = (0..size * size).map(|i| tile[(i / size % 32) * 32 + i % 32]).collect();
        let noise: Vec<f32> = (0..size * size).map(|_| rng.f32()).collect();
        let r = tiling_score(&repeated, size, 32).unwrap_or_else(|e| panic!("{e}"));
        let n = tiling_score(&noise, size, 32).unwrap_or_else(|e| panic!("{e}"));
        assert!(r > 0.99, "{r}");
        assert!(n < 0.1, "{n}");
        assert!(tiling_score(&noise, size, 0).is_err());
        assert!(autocorrelation(&vec![1.0; 64], 8).is_err());
    }
}
