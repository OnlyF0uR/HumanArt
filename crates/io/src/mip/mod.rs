//! Mip chain generation: the CPU reference of the GPU mip generator `[IO-05]`.
//!
//! The GPU generator (`humanart-render::compress::mips`) follows this module step for step and its
//! tests compare against it. Three content paths, chosen by the image's [`ColorSpace`]:
//!
//! - **Colour** (`Srgb`, `Linear`): sRGB samples are decoded to linear light before filtering and
//!   encoded after, so a mip of a black-and-white checker is 50 % grey in light (sRGB code 188), not
//!   code 128. With alpha (2 or 4 channels) the colour is filtered premultiplied by alpha and divided
//!   by the filtered alpha afterwards, so fully transparent texels do not bleed their colour.
//! - **Data** (`Data`): every channel is filtered independently, as stored.
//! - **Normal** (`Normal`): see [`generate_normal`]. The vectors are averaged, renormalised, and the
//!   length lost by averaging is turned into extra roughness (Toksvig-style, via a von Mises–Fisher
//!   fit), so a bumpy surface seen from afar keeps its specular energy spread instead of turning into
//!   a mirror.
//!
//! **The filter** is an exact area (box) filter for any size: level `k + 1` has
//! `max(1, ⌊size / 2⌋)` texels a side, and each of its texels averages the source area it covers,
//! with fractional weights at the edges when the source size is odd (at most 3 × 3 taps). The
//! weights are integer ratios, computed identically on the GPU.
//!
//! Errors are reported as [`Format::Ktx2`] errors: mip chains are made for KTX2 containers.
//!
//! **Precision:** every level is filtered from the previous level's `f32` working values, never
//! from its quantised output, so rounding does not accumulate down the chain (the GPU keeps an
//! `rgba32float` working chain for the same reason).

use humanart_math::color::transfer::{linear_to_srgb_f32, linear_to_srgb8, srgb_to_linear_f32, srgb8_to_linear};
use humanart_math::{f16_bits_to_f32, f32_to_f16_bits};

use crate::error::{Error, Format, Result};
use crate::image::{ColorSpace, Image, Samples};

/// The number of levels of a full chain for a `width × height` image (down to 1 × 1), 0 for an
/// empty size.
#[must_use]
pub const fn level_count(width: u32, height: u32) -> u32 {
    let side = if width > height { width } else { height };
    if side == 0 { 0 } else { 32 - side.leading_zeros() }
}

/// The size of level `level` of a `width × height` image: `max(1, size >> level)` per axis.
#[must_use]
pub const fn level_size(width: u32, height: u32, level: u32) -> (u32, u32) {
    let shift = if level > 31 { 31 } else { level };
    let w = width >> shift;
    let h = height >> shift;
    (if w == 0 { 1 } else { w }, if h == 0 { 1 } else { h })
}

/// The source taps of destination texel `x` when `src` texels are reduced to `dst` (with
/// `dst = max(1, src / 2)`): up to three `(source index, weight)` pairs whose weights sum to 1.
#[must_use]
pub fn taps(x: u32, src: u32, dst: u32) -> [(u32, f32); 3] {
    let mut out = [(0, 0.0); 3];
    if src == 0 || dst == 0 {
        return out;
    }
    let (lo, hi) = (u64::from(x) * u64::from(src), (u64::from(x) + 1) * u64::from(src));
    let d = u64::from(dst);
    let first = lo / d;
    for (k, slot) in out.iter_mut().enumerate() {
        let i = first + k as u64;
        let (a, b) = ((i * d).max(lo), ((i + 1) * d).min(hi));
        if b > a && i < u64::from(src) {
            *slot = (i as u32, (b - a) as f32 / src as f32);
        }
    }
    out
}

/// One level of `f32` working values: `channels` interleaved values per texel.
struct Work {
    width: u32,
    height: u32,
    channels: usize,
    data: Vec<f32>,
}

impl Work {
    /// The next level by the area filter.
    fn reduce(&self) -> Self {
        let (w, h) = level_size(self.width, self.height, 1);
        let c = self.channels;
        // Horizontal pass to `w × self.height`, then vertical to `w × h`.
        let mut mid = vec![0.0_f32; w as usize * self.height as usize * c];
        for y in 0..self.height as usize {
            for x in 0..w {
                let out = (y * w as usize + x as usize) * c;
                for (i, weight) in taps(x, self.width, w) {
                    if weight > 0.0 {
                        let src = (y * self.width as usize + i as usize) * c;
                        for k in 0..c {
                            mid[out + k] += weight * self.data[src + k];
                        }
                    }
                }
            }
        }
        let mut data = vec![0.0_f32; w as usize * h as usize * c];
        for y in 0..h {
            for (j, weight) in taps(y, self.height, h) {
                if weight > 0.0 {
                    for x in 0..w as usize {
                        let out = (y as usize * w as usize + x) * c;
                        let src = (j as usize * w as usize + x) * c;
                        for k in 0..c {
                            data[out + k] += weight * mid[src + k];
                        }
                    }
                }
            }
        }
        Self { width: w, height: h, channels: c, data }
    }
}

/// Reads sample `i` as a float: unsigned-normalised types map to `0..=1`.
fn sample(samples: &Samples, i: usize) -> f32 {
    match samples {
        Samples::U8(v) => f32::from(v[i]) / 255.0,
        Samples::U16(v) => f32::from(v[i]) / 65535.0,
        Samples::F16(v) => f16_bits_to_f32(v[i]),
        Samples::F32(v) => v[i],
    }
}

/// Empty sample storage of the same type as `like`, with room for `len` samples.
fn samples_like(like: &Samples, len: usize) -> Samples {
    match like {
        Samples::U8(_) => Samples::U8(Vec::with_capacity(len)),
        Samples::U16(_) => Samples::U16(Vec::with_capacity(len)),
        Samples::F16(_) => Samples::F16(Vec::with_capacity(len)),
        Samples::F32(_) => Samples::F32(Vec::with_capacity(len)),
    }
}

/// Appends `v` (`0..=1` for the normalised types) with rounding and clamping.
fn push(samples: &mut Samples, v: f32) {
    match samples {
        Samples::U8(out) => out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8),
        Samples::U16(out) => out.push((v.clamp(0.0, 1.0) * 65535.0).round() as u16),
        Samples::F16(out) => out.push(f32_to_f16_bits(v)),
        Samples::F32(out) => out.push(v),
    }
}

/// Whether channel `c` of a `channels`-channel image is alpha.
const fn is_alpha(channels: usize, c: usize) -> bool {
    (channels == 2 && c == 1) || (channels == 4 && c == 3)
}

/// Decodes a colour sample to linear light.
fn decode_color(samples: &Samples, i: usize, srgb: bool) -> f32 {
    match (samples, srgb) {
        (Samples::U8(v), true) => srgb8_to_linear(v[i]),
        (_, true) => srgb_to_linear_f32(sample(samples, i)),
        (_, false) => sample(samples, i),
    }
}

/// Appends a linear colour value, sRGB-encoded when `srgb`.
fn push_color(samples: &mut Samples, v: f32, srgb: bool) {
    match (samples, srgb) {
        (Samples::U8(out), true) => out.push(linear_to_srgb8(v)),
        (samples, true) => push(samples, linear_to_srgb_f32(v.clamp(0.0, 1.0))),
        (samples, false) => push(samples, v),
    }
}

/// Checks a level count request: 0 means the full chain.
fn levels_for(image: &Image, levels: u32) -> Result<u32> {
    let full = level_count(image.width, image.height);
    if levels > full {
        return Err(Error::Encode { format: Format::Ktx2, what: "more mip levels than the image size allows" });
    }
    if image.samples.len() != image.pixel_count() * image.channels as usize {
        return Err(Error::Encode { format: Format::Ktx2, what: "sample count does not match the dimensions" });
    }
    Ok(if levels == 0 { full } else { levels })
}

/// Generates `levels` mip levels of a colour or data image (0 = the full chain); level 0 is a copy
/// of `image`. Each level has the input's channels, sample type and colour space.
///
/// `Normal` images go through [`generate_normal`] with a roughness of 0 and only the normals are
/// returned.
pub fn generate(image: &Image, levels: u32) -> Result<Vec<Image>> {
    if image.color_space == ColorSpace::Normal {
        return generate_normal(image, None, 0.0, levels).map(|m| m.normals);
    }
    let levels = levels_for(image, levels)?;
    let channels = image.channels as usize;
    let srgb = image.color_space == ColorSpace::Srgb;
    let color = matches!(image.color_space, ColorSpace::Srgb | ColorSpace::Linear);
    let premultiply = color && (channels == 2 || channels == 4);
    let alpha_channel = channels - 1;

    let mut data = Vec::with_capacity(image.samples.len());
    for p in 0..image.pixel_count() {
        let base = p * channels;
        let alpha = if premultiply { sample(&image.samples, base + alpha_channel) } else { 1.0 };
        for c in 0..channels {
            data.push(if is_alpha(channels, c) && premultiply {
                alpha
            } else if color {
                decode_color(&image.samples, base + c, srgb) * alpha
            } else {
                sample(&image.samples, base + c)
            });
        }
    }
    let mut work = Work { width: image.width, height: image.height, channels, data };
    let mut out = vec![image.clone()];
    for _ in 1..levels {
        work = work.reduce();
        let mut samples = samples_like(&image.samples, work.data.len());
        for texel in work.data.chunks_exact(channels) {
            let alpha = if premultiply { texel[alpha_channel] } else { 1.0 };
            for (c, &v) in texel.iter().enumerate() {
                if is_alpha(channels, c) && premultiply {
                    push(&mut samples, v);
                } else if color {
                    let v = if premultiply { if alpha > 0.0 { v / alpha } else { 0.0 } } else { v };
                    push_color(&mut samples, v, srgb);
                } else {
                    push(&mut samples, v);
                }
            }
        }
        out.push(Image::new(Format::Ktx2, work.width, work.height, image.channels, image.color_space, samples)?);
    }
    Ok(out)
}

/// The mip chains of a normal map and its roughness.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalMips {
    /// The renormalised normals, one image per level, in the input's layout.
    pub normals: Vec<Image>,
    /// The adjusted perceptual roughness, one single-channel `Data` image per level, in the
    /// sample type of the roughness input (8-bit when there was none).
    pub roughness: Vec<Image>,
}

/// GGX `α²` from a perceptual roughness (`α = roughness²`).
fn alpha_sq(roughness: f32) -> f32 {
    let a = roughness.clamp(0.0, 1.0);
    a * a * a * a
}

/// The roughness `α²` a filtered normal of length `r` adds: `2 / κ` with the von Mises–Fisher
/// concentration `κ = (3r − r³) / (1 − r²)` (Banerjee et al.'s estimate), the Toksvig-style
/// mapping used by Neubelt and Pettineo ("Crafting a Next-Gen Material Pipeline for The Order:
/// 1886", 2013) and by Filament. `r ≥ 1` adds nothing; `r → 0` saturates at 1.
#[must_use]
pub fn variance_alpha_sq(r: f32) -> f32 {
    if r.is_nan() || r >= 1.0 {
        return 0.0;
    }
    let r = r.max(0.0);
    let kappa = (3.0 * r - r * r * r) / (1.0 - r * r);
    if kappa <= 0.0 { 1.0 } else { (2.0 / kappa).min(1.0) }
}

/// The perceptual roughness of a level from its mean `α²` and the mean normal length `r`:
/// `(min(1, mean α² + 2/κ))^(1/4)`.
#[must_use]
pub fn adjusted_roughness(mean_alpha_sq: f32, r: f32) -> f32 {
    (mean_alpha_sq + variance_alpha_sq(r)).clamp(0.0, 1.0).sqrt().sqrt()
}

/// Generates the mip chains of a normal map and its roughness (`[IO-05]` normal-aware filter).
///
/// `normals` holds 2, 3 or 4 channels: unsigned-normalised samples store `n · 0.5 + 0.5`, float
/// samples store `n`; with 2 channels `z = sqrt(1 − x² − y²)` is reconstructed; a fourth channel is
/// filtered as plain data. `roughness` is an optional single-channel (or first-channel) perceptual
/// roughness map of the same size; without one, `base_roughness` is used everywhere.
///
/// Method (variance-preserving): the working values are the unnormalised mean normal and the mean
/// GGX `α²`, both linear in the texels, so every level is the exact area mean of level 0. Each
/// output normal is the mean renormalised; the output roughness is
/// [`adjusted_roughness`]`(mean α², |mean normal|)`.
pub fn generate_normal(
    normals: &Image,
    roughness: Option<&Image>,
    base_roughness: f32,
    levels: u32,
) -> Result<NormalMips> {
    let levels = levels_for(normals, levels)?;
    let channels = normals.channels as usize;
    if channels < 2 {
        return Err(Error::Encode { format: Format::Ktx2, what: "a normal map needs at least 2 channels" });
    }
    if let Some(r) = roughness
        && (r.width != normals.width
            || r.height != normals.height
            || r.samples.len() != r.pixel_count() * r.channels as usize)
    {
        return Err(Error::Encode { format: Format::Ktx2, what: "the roughness map does not match the normal map" });
    }
    let unorm = matches!(normals.samples, Samples::U8(_) | Samples::U16(_));
    let unpack = |v: f32| if unorm { v * 2.0 - 1.0 } else { v };
    let pack = |v: f32| if unorm { v * 0.5 + 0.5 } else { v };
    // Working channels: x, y, z, α², then the plain fourth channel when present.
    let wc = if channels == 4 { 5 } else { 4 };
    let mut data = Vec::with_capacity(normals.pixel_count() * wc);
    for p in 0..normals.pixel_count() {
        let base = p * channels;
        let x = unpack(sample(&normals.samples, base));
        let y = unpack(sample(&normals.samples, base + 1));
        let z = if channels >= 3 {
            unpack(sample(&normals.samples, base + 2))
        } else {
            (1.0 - x * x - y * y).max(0.0).sqrt()
        };
        let len = (x * x + y * y + z * z).sqrt();
        let n = if len > 0.0 { [x / len, y / len, z / len] } else { [0.0, 0.0, 1.0] };
        let rough = roughness.map_or(base_roughness, |r| sample(&r.samples, p * r.channels as usize));
        data.extend_from_slice(&n);
        data.push(alpha_sq(rough));
        if channels == 4 {
            data.push(sample(&normals.samples, base + 3));
        }
    }
    let rough_like = roughness.map_or(Samples::U8(Vec::new()), |r| samples_like(&r.samples, 0));
    let mut work = Work { width: normals.width, height: normals.height, channels: wc, data };
    let mut out = NormalMips { normals: Vec::new(), roughness: Vec::new() };
    for level in 0..levels {
        if level > 0 {
            work = work.reduce();
        }
        let texels = work.data.len() / wc;
        let mut ns = samples_like(&normals.samples, texels * channels);
        let mut rs = samples_like(&rough_like, texels);
        for t in work.data.chunks_exact(wc) {
            let r = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            let n = if r > 0.0 { [t[0] / r, t[1] / r, t[2] / r] } else { [0.0, 0.0, 1.0] };
            for &v in n.iter().take(channels.min(3)) {
                push(&mut ns, pack(v));
            }
            if channels == 4 {
                push(&mut ns, t[4]);
            }
            push(&mut rs, adjusted_roughness(t[3], r));
        }
        out.normals.push(Image::new(Format::Ktx2, work.width, work.height, normals.channels, ColorSpace::Normal, ns)?);
        out.roughness.push(Image::new(Format::Ktx2, work.width, work.height, 1, ColorSpace::Data, rs)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    fn image(w: u32, h: u32, channels: u8, cs: ColorSpace, samples: Vec<u8>) -> Image {
        Image::new(Format::Ktx2, w, h, channels, cs, Samples::U8(samples)).expect("valid image")
    }

    #[test]
    fn level_sizes() {
        assert_eq!(level_count(4096, 4096), 13);
        assert_eq!(level_count(5, 3), 3);
        assert_eq!(level_count(1, 1), 1);
        assert_eq!(level_count(0, 7), 3);
        assert_eq!(level_size(5, 3, 1), (2, 1));
        assert_eq!(level_size(5, 3, 2), (1, 1));
        assert_eq!(level_size(5, 3, 40), (1, 1));
    }

    #[test]
    fn io_05_area_taps_cover_the_source_exactly() {
        for src in 1..40_u32 {
            let dst = (src / 2).max(1);
            let mut coverage = vec![0.0_f64; src as usize];
            for x in 0..dst {
                let t = taps(x, src, dst);
                let sum: f32 = t.iter().map(|&(_, w)| w).sum();
                assert!((sum - 1.0).abs() < 1e-6, "src {src} x {x}: {t:?}");
                for (i, w) in t {
                    coverage[i as usize] += f64::from(w) * f64::from(src) / f64::from(dst);
                }
            }
            for (i, c) in coverage.iter().enumerate() {
                assert!((c - 1.0).abs() < 1e-5, "src {src}: texel {i} covered {c}");
            }
        }
        assert_eq!(taps(0, 4, 2), [(0, 0.5), (1, 0.5), (0, 0.0)]);
    }

    /// A black/white checker averages to half the light: sRGB code 188, not 128.
    #[test]
    fn io_05_srgb_checker_filters_in_linear_light() {
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                px.push(if (x + y) % 2 == 0 { 0 } else { 255 });
            }
        }
        let srgb = generate(&image(4, 4, 1, ColorSpace::Srgb, px.clone()), 0).expect("mips");
        assert_eq!(srgb.len(), 3);
        assert_eq!(srgb[1].as_u8(), Some(&[188_u8; 4][..]));
        assert_eq!(srgb[2].as_u8(), Some(&[188_u8][..]));
        let data = generate(&image(4, 4, 1, ColorSpace::Data, px), 0).expect("mips");
        assert_eq!(data[1].as_u8(), Some(&[128_u8; 4][..]));
    }

    /// A transparent red texel next to an opaque blue one gives pure blue at half alpha.
    #[test]
    fn io_05_premultiplied_alpha_does_not_bleed() {
        let px = vec![255, 0, 0, 0, 0, 0, 255, 255];
        let mips = generate(&image(2, 1, 4, ColorSpace::Srgb, px), 0).expect("mips");
        assert_eq!(mips[1].as_u8(), Some(&[0_u8, 0, 255, 128][..]));
    }

    #[test]
    fn io_05_odd_sizes_and_float_samples() {
        let mut rng = Rng::new(5);
        let samples: Vec<f32> = (0..7 * 5 * 3).map(|_| rng.f32()).collect();
        let img = Image::new(Format::Ktx2, 7, 5, 3, ColorSpace::Linear, Samples::F32(samples.clone())).expect("ok");
        let mips = generate(&img, 0).expect("mips");
        assert_eq!(mips.iter().map(|m| (m.width, m.height)).collect::<Vec<_>>(), [(7, 5), (3, 2), (1, 1)]);
        // The 1x1 level is the mean of everything (area filter composes exactly).
        let mean: f32 = samples.as_chunks::<3>().0.iter().map(|p| p[0]).sum::<f32>() / 35.0;
        let last = mips[2].as_f32().expect("f32")[0];
        assert!((last - mean).abs() < 1e-5, "{last} vs {mean}");
        assert!(generate(&img, 4).is_err(), "too many levels");
    }

    /// A flat normal map keeps its normals and roughness; a bumpy one gets rougher and stays unit.
    #[test]
    fn io_05_normal_mips_renormalise_and_add_roughness() {
        let flat = image(4, 4, 3, ColorSpace::Normal, [128_u8, 128, 255].repeat(16));
        let mips = generate_normal(&flat, None, 0.4, 0).expect("mips");
        for (n, r) in mips.normals.iter().zip(&mips.roughness) {
            assert!(n.as_u8().expect("u8").as_chunks::<3>().0.iter().all(|p| *p == [128, 128, 255]));
            assert!(r.as_u8().expect("u8").iter().all(|&v| v == 102), "{:?}", r.as_u8()); // 0.4 * 255
        }
        // Alternating tilts of ±45° about y.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let sx = if (x + y) % 2 == 0 { s } else { -s };
                px.extend([sx, 0.0, s]);
            }
        }
        let bumpy = Image::new(Format::Ktx2, 4, 4, 3, ColorSpace::Normal, Samples::F32(px)).expect("ok");
        let mips = generate_normal(&bumpy, None, 0.2, 0).expect("mips");
        let n1 = mips.normals[1].as_f32().expect("f32");
        assert!((n1[0]).abs() < 1e-6 && (n1[2] - 1.0).abs() < 1e-6, "{n1:?}");
        let r0 = mips.roughness[0].as_u8().expect("u8")[0];
        let r1 = mips.roughness[1].as_u8().expect("u8")[0];
        // |mean| = 1/sqrt(2): alpha² = 0.2^4 + 2/kappa.
        let r = s;
        let expected = adjusted_roughness(alpha_sq(0.2), r);
        assert_eq!(r0, 51);
        assert_eq!(r1, (expected * 255.0).round() as u8);
        assert!(r1 > 150, "a ±45° bump field is rough from afar: {r1}");
    }

    /// Variance preservation: computing level 2 directly from level 0 (mean of 16 normals and α²)
    /// gives the same roughness as the chain.
    #[test]
    fn io_05_normal_chain_is_the_exact_area_mean() {
        let mut rng = Rng::new(77);
        let mut px = Vec::new();
        let mut rough = Vec::new();
        for _ in 0..16 {
            let (x, y) = (rng.range_f32(-0.6, 0.6), rng.range_f32(-0.6, 0.6));
            px.extend([x, y]);
            rough.push(rng.f32());
        }
        let normals = Image::new(Format::Ktx2, 4, 4, 2, ColorSpace::Normal, Samples::F32(px.clone())).expect("ok");
        let r = Image::new(Format::Ktx2, 4, 4, 1, ColorSpace::Data, Samples::F32(rough.clone())).expect("ok");
        let mips = generate_normal(&normals, Some(&r), 0.0, 0).expect("mips");
        let mut mean = [0.0_f32; 3];
        let mut a2 = 0.0;
        for (p, &ro) in px.as_chunks::<2>().0.iter().zip(&rough) {
            let z = (1.0 - p[0] * p[0] - p[1] * p[1]).sqrt();
            mean[0] += p[0] / 16.0;
            mean[1] += p[1] / 16.0;
            mean[2] += z / 16.0;
            a2 += alpha_sq(ro) / 16.0;
        }
        let len = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        let got = mips.roughness[2].as_f32().expect("f32")[0];
        assert!((got - adjusted_roughness(a2, len)).abs() < 1e-5);
        let n = mips.normals[2].as_f32().expect("f32");
        assert!((n[0] - mean[0] / len).abs() < 1e-5 && (n[1] - mean[1] / len).abs() < 1e-5);
    }

    #[test]
    fn variance_mapping_limits() {
        assert_eq!(variance_alpha_sq(1.0), 0.0);
        assert_eq!(variance_alpha_sq(0.0), 1.0);
        assert!(variance_alpha_sq(0.999) < 0.003);
        assert_eq!(variance_alpha_sq(f32::NAN), 0.0);
        assert!(adjusted_roughness(0.0, 1.0) == 0.0 && adjusted_roughness(1.0, 0.5) == 1.0);
    }
}
