//! The five scanline filters of PNG (ISO/IEC 15948, section 9) and the encoder's per-row choice.
//!
//! A filter predicts each byte from the byte one pixel to the left (`a`), the byte above (`b`) and
//! the byte above-left (`c`), and stores the difference modulo 256. The decoder's inverse runs over
//! fixed-size pixel chunks (`bpp` of 1, 2, 3, 4, 6 or 8 bytes) so the inner loops carry no bounds
//! checks.

/// Filter type 0: the bytes as they are.
pub(super) const NONE: u8 = 0;
/// Filter type 1: difference to the left neighbour.
pub(super) const SUB: u8 = 1;
/// Filter type 2: difference to the byte above.
pub(super) const UP: u8 = 2;
/// Filter type 3: difference to the mean of left and above.
pub(super) const AVERAGE: u8 = 3;
/// Filter type 4: difference to the Paeth predictor.
pub(super) const PAETH: u8 = 4;

/// The Paeth predictor: whichever of `a`, `b`, `c` is closest to `a + b - c` (ties in that order).
#[inline(always)]
pub(super) fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a16, b16, c16) = (i16::from(a), i16::from(b), i16::from(c));
    let pa = (b16 - c16).abs();
    let pb = (a16 - c16).abs();
    let pc = (a16 + b16 - 2 * c16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverses filter `kind` on `row` in place, `N` bytes per pixel. `prev` is the unfiltered row above
/// (all zeros for a pass's first row) and as long as `row`, whose length is a multiple of `N`.
#[inline(always)]
fn unfilter_n<const N: usize>(kind: u8, row: &mut [u8], prev: &[u8]) {
    let pixels = row.as_chunks_mut::<N>().0;
    let above = prev.as_chunks::<N>().0;
    match kind {
        SUB => {
            let mut a = [0_u8; N];
            for px in pixels {
                for k in 0..N {
                    px[k] = px[k].wrapping_add(a[k]);
                }
                a = *px;
            }
        }
        UP => {
            for (px, b) in pixels.iter_mut().zip(above) {
                for k in 0..N {
                    px[k] = px[k].wrapping_add(b[k]);
                }
            }
        }
        AVERAGE => {
            let mut a = [0_u8; N];
            for (px, b) in pixels.iter_mut().zip(above) {
                for k in 0..N {
                    px[k] = px[k].wrapping_add(a[k].midpoint(b[k]));
                }
                a = *px;
            }
        }
        PAETH => {
            let (mut a, mut c) = ([0_u8; N], [0_u8; N]);
            for (px, b) in pixels.iter_mut().zip(above) {
                for k in 0..N {
                    px[k] = px[k].wrapping_add(paeth(a[k], b[k], c[k]));
                }
                a = *px;
                c = *b;
            }
        }
        _ => {}
    }
}

/// Reverses filter `kind` on `row` in place; `prev` is the unfiltered row above (zeros for a
/// pass's first row), `bpp` the bytes per complete pixel (at least 1). Returns `false` for an
/// unknown filter type or a `bpp` no PNG has.
pub(super) fn unfilter(kind: u8, row: &mut [u8], prev: &[u8], bpp: usize) -> bool {
    if kind > PAETH || prev.len() != row.len() || !row.len().is_multiple_of(bpp.max(1)) {
        return false;
    }
    if kind == NONE {
        return true;
    }
    match bpp {
        1 => unfilter_n::<1>(kind, row, prev),
        2 => unfilter_n::<2>(kind, row, prev),
        3 => unfilter_n::<3>(kind, row, prev),
        4 => unfilter_n::<4>(kind, row, prev),
        6 => unfilter_n::<6>(kind, row, prev),
        8 => unfilter_n::<8>(kind, row, prev),
        _ => return false,
    }
    true
}

/// Applies filter `kind` to `row` (with `prev` the row above, zeros for the first) into `out`, all
/// three the same length.
pub(super) fn filter(kind: u8, row: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) {
    for (i, ((o, &x), &b)) in out.iter_mut().zip(row).zip(prev).enumerate() {
        let a = if i >= bpp { row[i - bpp] } else { 0 };
        let c = if i >= bpp { prev[i - bpp] } else { 0 };
        let predicted = match kind {
            SUB => a,
            UP => b,
            AVERAGE => a.midpoint(b),
            PAETH => paeth(a, b, c),
            _ => 0,
        };
        *o = x.wrapping_sub(predicted);
    }
}

/// The cost the adaptive choice minimises: the sum of the filtered bytes read as signed values
/// (the heuristic of the PNG specification, section 12.8).
pub(super) fn cost(filtered: &[u8]) -> u64 {
    filtered.iter().map(|&v| u64::from((v as i8).unsigned_abs())).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paeth_matches_the_specification() {
        // The specification's definition, with the ties broken a, then b, then c.
        for a in (0..=255_u8).step_by(15) {
            for b in (0..=255_u8).step_by(17) {
                for c in (0..=255_u8).step_by(13) {
                    let p = i32::from(a) + i32::from(b) - i32::from(c);
                    let (pa, pb, pc) = ((p - i32::from(a)).abs(), (p - i32::from(b)).abs(), (p - i32::from(c)).abs());
                    let expected = if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    };
                    assert_eq!(paeth(a, b, c), expected, "{a} {b} {c}");
                }
            }
        }
    }

    #[test]
    fn every_filter_is_inverted_for_every_pixel_size() {
        let mut rng = humanart_core::rng::Rng::new(77);
        for bpp in [1_usize, 2, 3, 4, 6, 8] {
            let len = bpp * 9;
            let mut prev = vec![0_u8; len];
            rng.fill(&mut prev);
            let mut row = vec![0_u8; len];
            rng.fill(&mut row);
            for kind in NONE..=PAETH {
                let mut filtered = vec![0_u8; len];
                filter(kind, &row, &prev, bpp, &mut filtered);
                assert!(unfilter(kind, &mut filtered, &prev, bpp));
                assert_eq!(filtered, row, "bpp {bpp}, filter {kind}");
            }
        }
        let mut row = [0_u8; 6];
        assert!(!unfilter(5, &mut row, &[0; 6], 3), "unknown filter type");
        assert!(!unfilter(1, &mut row, &[0; 5], 3), "rows of different lengths");
        assert!(!unfilter(1, &mut row, &[0; 6], 4), "row not a whole number of pixels");
        assert!(!unfilter(1, &mut [0_u8; 10], &[0; 10], 5), "no PNG has 5-byte pixels");
        assert_eq!(cost(&[0, 1, 255, 128]), 1 + 1 + 128);
    }
}
