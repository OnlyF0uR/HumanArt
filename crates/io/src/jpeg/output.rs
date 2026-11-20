//! From decoded component planes to an [`Image`]: chroma upsampling, colour conversion to RGB, and
//! the EXIF orientation.
//!
//! **Upsampling** is "fancy" (triangle-filtered), as libjpeg's default: each output sample is the
//! linear interpolation of the two nearest component samples, with sample centres aligned the way
//! T.81 A.1.1 places them (`x_component = (x + 0.5) * h / hmax - 0.5`), clamped at the edges. For
//! the usual factor of 2 this is libjpeg's 3/4 + 1/4 filter; it works the same for every other ratio
//! (4:1:1, 4:4:0, and fractional ones such as 3:2). Weights are in 1/256, vertical first.
//!
//! **Colour conversion** (all exact fixed-point, as libjpeg's `jdcolor.c`): YCbCr by the JFIF
//! equations (BT.601 full range); RGB as is; CMYK and YCCK as Adobe writes them (inverted, so
//! `R = C * K / 255`, YCCK first going through YCbCr to the inverted C, M, Y). The result is sRGB.

use super::decoder::{Component, Frame};
use super::{ColorModel, FORMAT};
use crate::error::Result;
use crate::image::{ColorSpace, Image, Limits, Samples};

/// The two source samples and the weight (in 1/256) of the second, for output position `i` of a
/// component sampled `factor / max` as densely as the image, `len` samples long.
#[inline]
fn taps(i: usize, factor: usize, max: usize, len: usize) -> (usize, usize, u32) {
    // Position = ((2i + 1) * factor - max) / (2 * max).
    let n = (2 * i + 1) * factor;
    if n <= max {
        return (0, 0, 0);
    }
    let (n, den) = (n - max, 2 * max);
    let i0 = n / den;
    if i0 + 1 >= len {
        let last = len.saturating_sub(1);
        return (last, last, 0);
    }
    (i0, i0 + 1, (((n % den) * 256 + max) / den) as u32)
}

/// Upsamples one component row by row to the image size.
struct Upsampler {
    /// Whether the component is already at full resolution.
    full: bool,
    /// Whether it has exactly half the horizontal resolution (the fast path).
    double: bool,
    /// Per output column: the two source columns and the weight of the second.
    cols: Vec<(u32, u32, u32)>,
    /// The vertically interpolated row, in 1/256 (on the fast path with one edge sample repeated
    /// at each end).
    mid: Vec<u16>,
    /// The output row.
    row: Vec<u8>,
}

impl Upsampler {
    fn new(c: &Component, frame: &Frame) -> Self {
        let full = c.h == frame.hmax && c.v == frame.vmax;
        let cols = if full {
            Vec::new()
        } else {
            (0..frame.width)
                .map(|x| {
                    let (a, b, w) = taps(x, c.h, frame.hmax, c.width);
                    (a as u32, b as u32, w)
                })
                .collect()
        };
        let double = c.h * 2 == frame.hmax;
        let (mid, row) = if full {
            (Vec::new(), Vec::new())
        } else {
            (vec![0; c.width + 2 * usize::from(double)], vec![0; frame.width])
        };
        Self { full, double, cols, mid, row }
    }

    /// Output row `y` of component `c`.
    fn row<'s>(&'s mut self, c: &'s Component, frame: &Frame, y: usize) -> &'s [u8] {
        let stride = c.stride();
        if self.full {
            return &c.plane[y * stride..y * stride + frame.width];
        }
        let (r0, r1, wy) = taps(y, c.v, frame.vmax, c.height);
        let a = &c.plane[r0 * stride..r0 * stride + c.width];
        let b = &c.plane[r1 * stride..r1 * stride + c.width];
        let (wa, wb) = (256 - wy as u16, wy as u16);
        let skip = usize::from(self.double);
        for ((m, &a), &b) in self.mid.iter_mut().skip(skip).zip(a).zip(b) {
            *m = u16::from(a) * wa + u16::from(b) * wb;
        }
        if self.double {
            // Factor 2, the common case: the taps are 1/4 and 3/4 of the neighbours, so
            // (64 a + 192 b + 32768) >> 16 = (a + 3 b + 512) >> 10. Repeating the edge samples
            // gives the edge clamp without a branch, so the loop vectorises.
            let n = self.mid.len();
            if n >= 3 {
                self.mid[0] = self.mid[1];
                self.mid[n - 1] = self.mid[n - 2];
            }
            let mid = &self.mid;
            let (pairs, rest) = self.row.as_chunks_mut::<2>();
            for (((pair, &a), &b), &c) in pairs.iter_mut().zip(mid).zip(mid.iter().skip(1)).zip(mid.iter().skip(2)) {
                let (a, b, c) = (u32::from(a), u32::from(b), u32::from(c));
                *pair = [((a + 3 * b + 512) >> 10) as u8, ((3 * b + c + 512) >> 10) as u8];
            }
            // An odd width ends with the first half of one more pair.
            if let (Some(o), Some(&a), Some(&b)) = (rest.first_mut(), mid.get(pairs.len()), mid.get(pairs.len() + 1)) {
                *o = ((u32::from(a) + 3 * u32::from(b) + 512) >> 10) as u8;
            }
            return &self.row;
        }
        let mid = &self.mid;
        for (o, &(i0, i1, wx)) in self.row.iter_mut().zip(&self.cols) {
            let s = u32::from(mid[i0 as usize]) * (256 - wx) + u32::from(mid[i1 as usize]) * wx;
            *o = ((s + 32768) >> 16) as u8;
        }
        &self.row
    }
}

/// Saturates to a byte.
#[inline(always)]
fn clamp(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// `round(a * b / 255)` for bytes, exactly.
#[inline(always)]
fn mul255(a: u8, b: u8) -> u8 {
    let x = u32::from(a) * u32::from(b) + 128;
    ((x + (x >> 8)) >> 8) as u8
}

/// JFIF YCbCr to RGB with libjpeg's 16-bit fixed-point constants and rounding.
#[inline(always)]
fn ycc_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (i32::from(y), i32::from(cb) - 128, i32::from(cr) - 128);
    [
        clamp(y + ((91_881 * cr + 32_768) >> 16)),
        clamp(y + ((-22_554 * cb - 46_802 * cr + 32_768) >> 16)),
        clamp(y + ((116_130 * cb + 32_768) >> 16)),
    ]
}

/// Pixels per batch of [`ycc_row`].
const BATCH: usize = 16;

/// Converts one row of YCbCr to interleaved RGB. Batches of [`BATCH`] pixels are converted into
/// planar arrays first (a loop the compiler vectorises) and interleaved after; the remainder pixel
/// by pixel. Same results as [`ycc_to_rgb`].
fn ycc_row(line: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    let (out, out_rest) = line.as_chunks_mut::<{ 3 * BATCH }>();
    let ((ys, y_rest), (cbs, cb_rest), (crs, cr_rest)) =
        (y.as_chunks::<BATCH>(), cb.as_chunks::<BATCH>(), cr.as_chunks::<BATCH>());
    for (((px, ys), cbs), crs) in out.iter_mut().zip(ys).zip(cbs).zip(crs) {
        let mut rgb = [[0_u8; BATCH]; 3];
        for i in 0..BATCH {
            let (y, cb, cr) = (i32::from(ys[i]), i32::from(cbs[i]) - 128, i32::from(crs[i]) - 128);
            rgb[0][i] = clamp(y + ((91_881 * cr + 32_768) >> 16));
            rgb[1][i] = clamp(y + ((-22_554 * cb - 46_802 * cr + 32_768) >> 16));
            rgb[2][i] = clamp(y + ((116_130 * cb + 32_768) >> 16));
        }
        for (i, p) in px.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *p = [rgb[0][i], rgb[1][i], rgb[2][i]];
        }
    }
    for (((px, &y), &cb), &cr) in out_rest.as_chunks_mut::<3>().0.iter_mut().zip(y_rest).zip(cb_rest).zip(cr_rest) {
        *px = ycc_to_rgb(y, cb, cr);
    }
}

/// Converts the decoded planes to an sRGB image: 1 channel for grey, 3 for everything else.
pub(super) fn to_image(frame: &Frame, model: ColorModel, limits: &Limits) -> Result<Image> {
    let channels = if model == ColorModel::Grey { 1 } else { 3 };
    let (w, h) = (frame.width, frame.height);
    let len = limits.check_bytes(FORMAT, "decoded image", (w * h * channels) as u64)?;
    let mut out = vec![0_u8; len];
    let mut ups: Vec<Upsampler> = frame.components.iter().map(|c| Upsampler::new(c, frame)).collect();
    for (y, line) in out.chunks_exact_mut(w * channels).enumerate() {
        let mut rows: [&[u8]; 4] = [&[]; 4];
        for ((r, up), c) in rows.iter_mut().zip(ups.iter_mut()).zip(&frame.components) {
            *r = up.row(c, frame, y);
        }
        match model {
            ColorModel::Grey => line.copy_from_slice(rows[0]),
            ColorModel::Rgb => {
                for (x, px) in line.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    px.copy_from_slice(&[rows[0][x], rows[1][x], rows[2][x]]);
                }
            }
            ColorModel::YCbCr => ycc_row(line, rows[0], rows[1], rows[2]),
            ColorModel::Cmyk => {
                for (x, px) in line.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let k = rows[3][x];
                    px.copy_from_slice(&[mul255(rows[0][x], k), mul255(rows[1][x], k), mul255(rows[2][x], k)]);
                }
            }
            ColorModel::Ycck => {
                for (x, px) in line.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let k = rows[3][x];
                    let [r, g, b] = ycc_to_rgb(rows[0][x], rows[1][x], rows[2][x]);
                    px.copy_from_slice(&[mul255(r, k), mul255(g, k), mul255(b, k)]);
                }
            }
        }
    }
    Image::new(FORMAT, w as u32, h as u32, channels as u8, ColorSpace::Srgb, Samples::U8(out))
}

/// Applies an EXIF orientation (1 to 8; anything else leaves the image as it is) so the result is
/// upright: 2 mirrors, 3 rotates 180°, 4 flips, 5 transposes, 6 rotates 90° clockwise, 7 transverses,
/// 8 rotates 90° counter-clockwise.
pub(super) fn orient(image: Image, orientation: u8) -> Result<Image> {
    let Some(src) = image.as_u8() else { return Ok(image) };
    if !(2..=8).contains(&orientation) {
        return Ok(image);
    }
    let (w, h, c) = (image.width as usize, image.height as usize, usize::from(image.channels));
    let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
    let mut out = vec![0_u8; src.len()];
    for (oy, line) in out.chunks_exact_mut(ow * c).enumerate() {
        for (ox, px) in line.chunks_exact_mut(c).enumerate() {
            let (sx, sy) = match orientation {
                2 => (w - 1 - ox, oy),
                3 => (w - 1 - ox, h - 1 - oy),
                4 => (ox, h - 1 - oy),
                5 => (oy, ox),
                6 => (oy, h - 1 - ox),
                7 => (w - 1 - oy, h - 1 - ox),
                _ => (w - 1 - oy, ox),
            };
            let at = (sy * w + sx) * c;
            px.copy_from_slice(&src[at..at + c]);
        }
    }
    Image::new(FORMAT, ow as u32, oh as u32, image.channels, image.color_space, Samples::U8(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_reproduce_the_libjpeg_triangle_filter() {
        // Factor 2: edge sample, then 1/4 and 3/4 steps.
        let got: Vec<_> = (0..6).map(|x| taps(x, 1, 2, 3)).collect();
        assert_eq!(got, [(0, 0, 0), (0, 1, 64), (0, 1, 192), (1, 2, 64), (1, 2, 192), (2, 2, 0)]);
        // Full resolution is the identity; factor 4 has 1/8 steps.
        assert!((0..9).all(|x| taps(x, 2, 2, 9).0 == x && taps(x, 2, 2, 9).2 == 0));
        assert_eq!(taps(2, 1, 4, 2), (0, 1, 32));
        assert_eq!(taps(5, 1, 4, 2), (0, 1, 224));
        // A fractional ratio 2:3 and a one-sample component.
        assert_eq!(taps(1, 2, 3, 4), (0, 1, 128));
        assert_eq!(taps(2, 2, 3, 4), (1, 2, 43));
        assert_eq!(taps(7, 1, 2, 1), (0, 0, 0));
    }

    #[test]
    fn colour_conversion_matches_the_float_equations() {
        for (y, cb, cr) in
            [(0_u8, 128_u8, 128_u8), (255, 128, 128), (76, 85, 255), (150, 44, 21), (29, 255, 107), (128, 0, 255)]
        {
            let (fy, fb, fr) = (f64::from(y), f64::from(cb) - 128.0, f64::from(cr) - 128.0);
            let expected = [fy + 1.402 * fr, fy - 0.344_136 * fb - 0.714_136 * fr, fy + 1.772 * fb];
            for (got, e) in ycc_to_rgb(y, cb, cr).iter().zip(expected) {
                assert!((f64::from(*got) - e.clamp(0.0, 255.0)).abs() <= 0.5 + 1e-3, "{y} {cb} {cr}");
            }
        }
        for a in 0..=255_u8 {
            for b in [0_u8, 1, 127, 128, 254, 255] {
                assert_eq!(mul255(a, b), ((f64::from(a) * f64::from(b)) / 255.0).round() as u8);
            }
        }
    }

    #[test]
    fn every_orientation_moves_the_corners_where_exif_says() {
        type Case = (u8, (u32, u32), [[u8; 2]; 3]);
        // A 3×2 RGB image whose pixel (x, y) is [x, y, 0].
        let samples: Vec<u8> = (0..2).flat_map(|y| (0..3).flat_map(move |x| [x, y, 0])).collect();
        let image = Image::new(FORMAT, 3, 2, 3, ColorSpace::Srgb, Samples::U8(samples)).unwrap();
        // The source pixel at each corner of the upright output: top-left, top-right, bottom-left.
        let cases: [Case; 8] = [
            (1, (3, 2), [[0, 0], [2, 0], [0, 1]]),
            (2, (3, 2), [[2, 0], [0, 0], [2, 1]]),
            (3, (3, 2), [[2, 1], [0, 1], [2, 0]]),
            (4, (3, 2), [[0, 1], [2, 1], [0, 0]]),
            (5, (2, 3), [[0, 0], [0, 1], [2, 0]]),
            (6, (2, 3), [[0, 1], [0, 0], [2, 1]]),
            (7, (2, 3), [[2, 1], [2, 0], [0, 1]]),
            (8, (2, 3), [[2, 0], [2, 1], [0, 0]]),
        ];
        for (o, (w, h), corners) in cases {
            let out = orient(image.clone(), o).unwrap();
            assert_eq!((out.width, out.height), (w, h), "orientation {o}");
            let s = out.as_u8().unwrap();
            let px = |x: u32, y: u32| {
                let i = ((y * w + x) * 3) as usize;
                [s[i], s[i + 1]]
            };
            assert_eq!([px(0, 0), px(w - 1, 0), px(0, h - 1)], corners, "orientation {o}");
        }
    }
}
