//! Seam-padding dilation `[BAKE-03]` and its verification through the mip chain `[AC-WP15-1]`.
//!
//! A baked map holds values only where the surface is (texel centres inside a UV triangle). A
//! filtered lookup near an island edge, and every texel of a coarser mip level whose footprint
//! straddles the edge, also reads the texels outside: left empty they pull the background into the
//! surface and show as a seam line that thickens with distance. Dilation grows every island
//! outward one ring of texels per pass:
//!
//! - [`DilateMode::Average`]: an empty texel with at least one filled 8-neighbour takes the mean of
//!   its filled neighbours, summed in the fixed order left, right, up, down, up-left, up-right,
//!   down-left, down-right, times [`RECIPROCALS`] (for continuous data: thickness, occlusion,
//!   curvature, normals);
//! - [`DilateMode::Nearest`]: it copies the first filled neighbour in the same order (for
//!   categorical data: region ids).
//!
//! Each pass reads only the previous pass's result, so the outcome does not depend on the texel
//! visiting order, and `shaders/bake/dilate.wgsl` produces bit-identical results.
//!
//! **How much padding:** a bilinear lookup at mip level `k` reads level-`k` texels up to one texel
//! beyond the one containing the sample, and each covers `2ᵏ` level-0 texels, so a lookup on the
//! surface is clean when the islands are padded by [`padding_for_levels`]`(k + 1) = 2ᵏ⁺¹` texels
//! (16 for levels 0 to 3). [`seam_leaks`] counts the texels for which this does not hold, level by
//! level; the islands themselves must be packed with a gutter of at least twice the padding for
//! neighbouring islands not to meet inside it.

use humanart_io::mip::{level_size, taps};

use super::{BakeError, BakeMap, Coverage, Result};

/// The most dilation passes.
pub const MAX_PASSES: u32 = 256;

/// How an empty texel is filled from its neighbours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DilateMode {
    /// The mean of the filled neighbours.
    Average,
    /// The first filled neighbour (left, right, up, down, then the diagonals).
    Nearest,
}

/// The neighbour offsets in the fixed summation order.
pub const NEIGHBOURS: [(i32, i32); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];

/// `1 / n` for `n` filled neighbours, as literals: the mean is `sum · RECIPROCALS[n − 1]` on the
/// CPU and on the GPU alike (a GPU division need not be correctly rounded, a product is).
pub const RECIPROCALS: [f32; 8] = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 1.0 / 7.0, 0.125];

/// The padding in texels that keeps bilinear lookups on the surface clean at mip levels
/// `0..levels`: `2^levels` (at least 1).
pub const fn padding_for_levels(levels: u32) -> u32 {
    if levels == 0 {
        1
    } else if levels >= 8 {
        MAX_PASSES
    } else {
        1 << levels
    }
}

/// Grows the filled texels of `map` outward by `passes` rings (see the module documentation).
/// New texels are marked [`Coverage::Padding`].
pub fn dilate(map: &mut BakeMap, passes: u32, mode: DilateMode) -> Result<()> {
    if passes > MAX_PASSES {
        return Err(BakeError::invalid("more than 256 dilation passes"));
    }
    let (w, h, c) = (map.width as i32, map.height as i32, map.channels as usize);
    if map.coverage.len() != map.texel_count() || map.data.len() != map.texel_count() * c {
        return Err(BakeError::invalid("the map's arrays do not match its size"));
    }
    let mut data = map.data.clone();
    let mut coverage = map.coverage.clone();
    for _ in 0..passes {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if map.coverage[i] != Coverage::Empty {
                    continue;
                }
                let mut sum = [0.0_f32; 4];
                let mut count = 0_u32;
                for (dx, dy) in NEIGHBOURS {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    if map.coverage[j] == Coverage::Empty {
                        continue;
                    }
                    let src = &map.data[j * c..j * c + c];
                    match mode {
                        DilateMode::Average => {
                            for (s, &v) in sum.iter_mut().zip(src) {
                                *s += v;
                            }
                            count += 1;
                        }
                        DilateMode::Nearest => {
                            sum[..c].copy_from_slice(src);
                            count = 1;
                            break;
                        }
                    }
                }
                if count > 0 {
                    let inv = RECIPROCALS[count as usize - 1];
                    for (d, s) in data[i * c..i * c + c].iter_mut().zip(&sum) {
                        *d = s * inv;
                    }
                    coverage[i] = Coverage::Padding;
                    changed = true;
                }
            }
        }
        map.data.copy_from_slice(&data);
        map.coverage.copy_from_slice(&coverage);
        if !changed {
            break;
        }
    }
    Ok(())
}

/// For each mip level `0..levels` of a map with this coverage (`width × height`, the chain of
/// `humanart_io::mip`): the number of level texels that a bilinear lookup at a surface point can
/// read (within one texel of a texel whose footprint holds a [`Coverage::Surface`] texel) but
/// whose footprint holds an [`Coverage::Empty`] texel. Zero at every level means no seam shows at
/// those levels `[AC-WP15-1]`.
pub fn seam_leaks(coverage: &[Coverage], width: u32, height: u32, levels: u32) -> Result<Vec<usize>> {
    if coverage.len() != width as usize * height as usize || width == 0 || height == 0 {
        return Err(BakeError::invalid("the coverage does not match the size"));
    }
    // Per level: does the footprint hold a surface texel, does it hold an empty one.
    let mut surface: Vec<bool> = coverage.iter().map(|&c| c == Coverage::Surface).collect();
    let mut empty: Vec<bool> = coverage.iter().map(|&c| c == Coverage::Empty).collect();
    let (mut sw, mut sh) = (width, height);
    let mut out = Vec::new();
    for level in 0..levels {
        if level > 0 {
            let (dw, dh) = level_size(width, height, level);
            let mut s2 = vec![false; dw as usize * dh as usize];
            let mut e2 = vec![false; dw as usize * dh as usize];
            for y in 0..dh {
                let ty = taps(y, sh, dh);
                for x in 0..dw {
                    let tx = taps(x, sw, dw);
                    let i = (y * dw + x) as usize;
                    for &(sy, wy) in &ty {
                        for &(sx, wx) in &tx {
                            if wy > 0.0 && wx > 0.0 {
                                let j = (sy * sw + sx) as usize;
                                s2[i] |= surface[j];
                                e2[i] |= empty[j];
                            }
                        }
                    }
                }
            }
            surface = s2;
            empty = e2;
            (sw, sh) = (dw, dh);
        }
        let mut leaks = 0;
        for y in 0..sh as i32 {
            for x in 0..sw as i32 {
                let i = (y * sw as i32 + x) as usize;
                if !empty[i] {
                    continue;
                }
                let reached = (-1..=1).any(|dy| {
                    (-1..=1).any(|dx| {
                        let (nx, ny) = (x + dx, y + dy);
                        nx >= 0
                            && ny >= 0
                            && nx < sw as i32
                            && ny < sh as i32
                            && surface[(ny * sw as i32 + nx) as usize]
                    })
                });
                leaks += usize::from(reached);
            }
        }
        out.push(leaks);
        if sw == 1 && sh == 1 {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::BakeTarget;

    /// Two discs of constant value 0.25 and 0.75 on an empty background.
    fn two_islands(size: u32) -> BakeMap {
        let mut map = BakeMap::new(&BakeTarget::new(0, size, size), 1, 0.0).unwrap();
        let s = size as f32;
        for y in 0..size {
            for x in 0..size {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let i = (y * size + x) as usize;
                if (fx - 0.3 * s).hypot(fy - 0.5 * s) < 0.17 * s {
                    map.set(i, &[0.25], Coverage::Surface);
                } else if (fx - 0.75 * s).hypot(fy - 0.4 * s) < 0.12 * s {
                    map.set(i, &[0.75], Coverage::Surface);
                }
            }
        }
        map
    }

    #[test]
    fn ac_wp15_1_padding_removes_leaks_at_mips_0_to_3() {
        let mut map = two_islands(256);
        let before = seam_leaks(&map.coverage, 256, 256, 4).unwrap();
        assert!(before.iter().all(|&l| l > 0), "{before:?}");
        dilate(&mut map, padding_for_levels(4), DilateMode::Average).unwrap();
        let after = seam_leaks(&map.coverage, 256, 256, 4).unwrap();
        assert_eq!(after, vec![0, 0, 0, 0]);
        // One pass short of the padding leaks at level 3.
        let mut short = two_islands(256);
        dilate(&mut short, padding_for_levels(4) / 2, DilateMode::Average).unwrap();
        assert!(seam_leaks(&short.coverage, 256, 256, 4).unwrap()[3] > 0);
    }

    #[test]
    fn ac_wp15_1_padded_constant_islands_keep_their_value_through_the_mips() {
        // The real mip generator of the exporter: every level-k texel a surface lookup can reach
        // holds exactly its island's constant.
        let mut map = two_islands(128);
        dilate(&mut map, padding_for_levels(4), DilateMode::Average).unwrap();
        let image = map.to_image(humanart_io::ColorSpace::Data).unwrap();
        let mips = humanart_io::mip::generate(&image, 4).unwrap();
        let surface0: Vec<bool> = map.coverage.iter().map(|&c| c == Coverage::Surface).collect();
        for (k, level) in mips.iter().enumerate() {
            let v = level.as_f32().unwrap();
            let step = 1 << k;
            for y in 0..level.height {
                for x in 0..level.width {
                    // A surface texel of level 0 inside this texel's footprint: its island value.
                    let mut island = None;
                    for sy in y * step..(y + 1) * step {
                        for sx in x * step..(x + 1) * step {
                            if surface0[(sy * 128 + sx) as usize] {
                                island = Some(map.data[(sy * 128 + sx) as usize]);
                            }
                        }
                    }
                    if let Some(want) = island {
                        let got = v[(y * level.width + x) as usize];
                        assert!((got - want).abs() < 1e-6, "level {k} ({x},{y}): {got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn nearest_mode_copies_ids_and_average_mode_averages() {
        let mut map = BakeMap::new(&BakeTarget::new(0, 3, 1), 1, 0.0).unwrap();
        map.set(0, &[2.0], Coverage::Surface);
        map.set(2, &[4.0], Coverage::Surface);
        let mut avg = map.clone();
        dilate(&mut avg, 1, DilateMode::Average).unwrap();
        assert_eq!(avg.data, vec![2.0, 3.0, 4.0]);
        assert_eq!(avg.coverage[1], Coverage::Padding);
        dilate(&mut map, 1, DilateMode::Nearest).unwrap();
        assert_eq!(map.data, vec![2.0, 2.0, 4.0]);
        assert!(dilate(&mut map, MAX_PASSES + 1, DilateMode::Average).is_err());
        assert!(seam_leaks(&map.coverage, 2, 2, 1).is_err());
    }
}
