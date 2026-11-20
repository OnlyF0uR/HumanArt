//! The LOD switching metric `[MOD-05]` `[RT-05]`: a character's projected size from its height,
//! its distance and the camera's vertical field of view, and the level choice with hysteresis
//! ([`crate::lod::select_lod`]).

use crate::lod::{LodLevel, select_lod};

/// The projected height of an object of `height` metres at `distance` metres from a perspective
/// camera with vertical field of view `fov_y` (radians), as a fraction of the screen height.
///
/// Degenerate input resolves towards detail: a distance that is not positive or not finite, or a
/// field of view outside `(0, π)`, gives `f32::INFINITY` (the finest level); a height that is not
/// positive gives 0.
#[must_use]
pub fn projected_size(height: f32, distance: f32, fov_y: f32) -> f32 {
    if height.is_nan() || height <= 0.0 {
        return 0.0;
    }
    if !distance.is_finite() || distance <= 0.0 || fov_y.is_nan() || fov_y <= 0.0 || fov_y >= std::f32::consts::PI {
        return f32::INFINITY;
    }
    height / (2.0 * distance * (0.5 * fov_y).tan())
}

/// The level for a character of `height` metres at `distance` metres seen with vertical field of
/// view `fov_y` while level `current` is shown: [`select_lod`] on [`projected_size`], so a level
/// switches coarser only below `switch_size * (1 - hysteresis)` and finer only above
/// `switch_size * (1 + hysteresis)`.
#[must_use]
pub fn choose_lod(levels: &[LodLevel], height: f32, distance: f32, fov_y: f32, current: usize) -> usize {
    select_lod(levels, projected_size(height, distance, fov_y), current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fixtures::grid;

    fn levels() -> Vec<LodLevel> {
        [0.5_f32, 0.25, 0.12, 0.06]
            .iter()
            .map(|&s| LodLevel {
                mesh: grid(1),
                sources: vec![[0, 0, 0]; 4],
                source_weights: vec![[1.0, 0.0, 0.0]; 4],
                declared_error: 0.0,
                switch_size: s,
                hysteresis: 0.1,
            })
            .collect()
    }

    #[test]
    fn mod_05_projected_size_follows_distance_and_fov() {
        let fov = 60_f32.to_radians();
        let s = projected_size(1.7, 2.0, fov);
        assert!((s - 1.7 / (4.0 * (30_f32.to_radians()).tan())).abs() < 1e-6);
        assert!((projected_size(1.7, 4.0, fov) - s / 2.0).abs() < 1e-6);
        assert!(projected_size(1.7, 2.0, fov / 2.0) > s);
        assert_eq!(projected_size(1.7, 0.0, fov), f32::INFINITY);
        assert_eq!(projected_size(1.7, f32::NAN, fov), f32::INFINITY);
        assert_eq!(projected_size(1.7, 2.0, 0.0), f32::INFINITY);
        assert_eq!(projected_size(1.7, 2.0, 4.0), f32::INFINITY);
        assert_eq!(projected_size(0.0, 2.0, fov), 0.0);
        assert_eq!(projected_size(f32::NAN, 2.0, fov), 0.0);
    }

    #[test]
    fn mod_05_switching_is_monotonic_with_hysteresis() {
        let l = levels();
        let fov = 50_f32.to_radians();
        // Walking away: the level never gets finer.
        let mut cur = 0;
        let mut d = 0.5;
        while d < 200.0 {
            let next = choose_lod(&l, 1.7, d, fov, cur);
            assert!(next >= cur, "at {d} m: {cur} -> {next}");
            cur = next;
            d *= 1.05;
        }
        assert_eq!(cur, 4);
        // Walking back: never coarser.
        while d > 0.5 {
            let next = choose_lod(&l, 1.7, d, fov, cur);
            assert!(next <= cur, "at {d} m: {cur} -> {next}");
            cur = next;
            d /= 1.05;
        }
        assert_eq!(cur, 0);
        // Jitter of ±5 % around a threshold does not flicker.
        let d0 = 1.7 / (2.0 * 0.25 * (0.5 * fov).tan());
        let mut cur = choose_lod(&l, 1.7, d0 * 1.2, fov, 1);
        assert_eq!(cur, 2);
        let mut switches = 0;
        for i in 0..100 {
            let d = d0 * if i % 2 == 0 { 0.95 } else { 1.05 };
            let next = choose_lod(&l, 1.7, d, fov, cur);
            switches += usize::from(next != cur);
            cur = next;
        }
        assert_eq!(switches, 0);
        // Degenerate distance: the finest level.
        assert_eq!(choose_lod(&l, 1.7, -1.0, fov, 3), 0);
    }
}
