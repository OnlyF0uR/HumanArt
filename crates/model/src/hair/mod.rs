//! The hair system: groom data, guides and interpolation, the CPU reference of the generation
//! modifiers, the brow / lash / facial / body hair generators, presets, colour, hair cards and the
//! `.anthair` file format (WP-17, `[HAIR-01]`..`[HAIR-06]`, `[HAIR-10]`, `[IO-12]`).
//!
//! - [`error`] holds [`HairError`].
//! - [`surface`]: [`HairSurface`] (the skin the hair grows from), [`RootFrame`] and the root-frame
//!   construction strands ride the deformed surface with, [`ProxyHead`] for tests.
//! - [`collide`]: closest-point queries on a triangle mesh, the signed distance grid the
//!   generator pushes strands out of the head with, and the penetration check `[AC-WP17-1]`.
//! - [`strands`]: [`Guides`] (edited, `f32`) and [`Strands`] (generated, `f16` offsets and the
//!   per-strand attributes) `[HAIR-01]`.
//! - [`params`]: [`GroomParams`] (every modifier, keyed for the document) and [`GroomMaps`].
//! - [`placement`]: [`ScalpMask`], [`place`] and [`GroomLayout`]: roots, guide weights, clumps.
//! - [`generate`]: the CPU reference of the modifiers and the attach step `[HAIR-02]`.
//! - [`presets`]: the preset grooms grown from guide templates `[HAIR-03]`.
//! - [`features`]: the brow and lash generators and the lid-following hook `[HAIR-04]`.
//! - [`facial`]: beard, moustache, goatee, stubble and body hair `[HAIR-05]`.
//! - [`settings`]: the document's character-wide groom settings and how they scale the groom set
//!   (todo 2.3).
//! - [`cards`]: hair-card geometry and atlas layout for game LODs `[HAIR-10]`.
//! - [`format`]: the `.anthair` strand file `[IO-12]`.
//! - [`color`]: melanin, grey and dye to absorption coefficients for the shader `[HAIR-06]`.
//! - [`template`]: the scalp mask, brow and lid lines and hair areas of the template, and the
//!   character's groom set grown on them ([`character_grooms`]).
//!
//! **Determinism.** Every random choice is drawn either from [`humanart_core::rng::Rng`] seeded by
//! the groom's seed (root placement, guide templates), or, inside the generation modifiers, from the
//! integer hash [`hash_u32`] of the strand key, which the WGSL modifiers compute bit-identically
//! (`shaders/groom/common.wgsl`), so the CPU reference and the GPU agree `[AC-WP17-2]`.
#![allow(clippy::doc_markdown)]

pub mod cards;
pub mod collide;
pub mod color;
pub mod error;
pub mod facial;
pub mod features;
pub mod format;
pub mod generate;
pub mod params;
pub mod placement;
pub mod presets;
pub mod settings;
pub mod strands;
pub mod surface;
pub mod template;

pub use collide::{ClosestPoint, PenetrationReport, SdfGrid, TriangleGrid};
pub use color::HairColor;
pub use error::{HairError, HairResult};
pub use facial::{BodyHairSpec, FaceAnchors, body_hair, facial_groom};
pub use features::{Eyeball, FeatureGroom, brow_groom, lash_groom, lift_lashes};
pub use generate::{GenerateInput, Generated, attach, generate, generate_groom, root_frames};
pub use params::{Cowlick, GroomMap, GroomMaps, GroomParams, Parting};
pub use placement::{GroomLayout, PartLine, ScalpMask, place, place_parted};
pub use presets::{PresetGroom, preset_groom};
pub use strands::{Guides, Strands};
pub use surface::{HairSurface, ProxyHead, RootFrame, root_frame};
pub use template::{CharacterGroom, GroomKind, GroomSetSpec, ScalpStyle, TemplateHair, character_grooms};

/// The PCG output hash of one 32-bit word (Jarzynski and Olano, "Hash Functions for GPU
/// Rendering", JCGT 2020). Mirrored by `groom_hash` in WGSL.
#[inline]
pub const fn hash_u32(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// The key of strand `strand` in a groom with seed `seed`: every per-strand random value is a hash
/// of the key with a salt. Mirrored by `groom_key` in WGSL.
#[inline]
pub const fn strand_key(seed: u32, strand: u32) -> u32 {
    hash_u32(seed ^ hash_u32(strand))
}

/// A uniform value in `[0, 1)` from `key` and `salt`, exact in `f32` (24 bits). Mirrored by
/// `groom_rand` in WGSL.
#[inline]
pub fn hash01(key: u32, salt: u32) -> f32 {
    (hash_u32(key ^ hash_u32(salt)) >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_the_pcg_output_function() {
        // Reference values of the published PCG hash, computed independently in Python.
        assert_eq!(hash_u32(0), 129_708_002);
        assert_eq!(hash_u32(1), 2_831_084_092);
        assert_eq!(hash_u32(12_345), 4_099_845_390);
        let mut seen = std::collections::HashSet::new();
        for i in 0..10_000 {
            seen.insert(hash_u32(i));
        }
        assert_eq!(seen.len(), 10_000);
        let mean: f32 = (0..10_000).map(|i| hash01(strand_key(7, i), 3)).sum::<f32>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        assert!((0..1000).all(|i| (0.0..1.0).contains(&hash01(i, i))));
    }
}

/// Contact sheets for the tests: strands drawn additively over a dotted surface, written to
/// `target/agents/hair/<name>` for a human (and the agent) to look at.
#[cfg(test)]
pub(crate) mod sheet {
    use humanart_math::Vec3;

    /// One panel of a sheet.
    pub struct Panel<'a> {
        /// Surface points drawn as dots.
        pub dots: &'a [Vec3],
        /// Strand points, `points` per strand.
        pub strands: &'a [Vec3],
        /// Points per strand.
        pub points: usize,
        /// The world point at the panel's anchor.
        pub centre: Vec3,
        /// Pixels per metre.
        pub scale: f32,
        /// Seen from the front (x right) or the character's left (z right).
        pub front: bool,
    }

    /// Draws `panels` in a grid of `cols` columns of `pw x ph` pixels and writes the PNG.
    pub fn write(name: &str, panels: &[Panel<'_>], cols: usize, pw: usize, ph: usize) {
        let rows = panels.len().div_ceil(cols);
        let (w, h) = (pw * cols, ph * rows);
        let mut rgb = vec![28_u8; w * h * 3];
        for (n, panel) in panels.iter().enumerate() {
            let (ox, oy) = ((n % cols) * pw, (n / cols) * ph);
            let project = |q: Vec3| {
                let v = q - panel.centre;
                let x = if panel.front { v.x } else { v.z };
                (ox as f32 + pw as f32 * 0.5 + x * panel.scale, oy as f32 + ph as f32 * 0.4 - v.y * panel.scale)
            };
            let mut put = |x: f32, y: f32, c: [u8; 3], add: bool| {
                let (x, y) = (x as i64, y as i64);
                if x >= ox as i64 && y >= oy as i64 && (x as usize) < ox + pw && (y as usize) < oy + ph {
                    let i = (y as usize * w + x as usize) * 3;
                    for k in 0..3 {
                        rgb[i + k] = if add { rgb[i + k].saturating_add(c[k]) } else { c[k] };
                    }
                }
            };
            for &q in panel.dots {
                let (x, y) = project(q);
                put(x, y, [150, 120, 105], false);
            }
            for strand in panel.strands.chunks(panel.points.max(1)) {
                for k in 1..strand.len() {
                    let (a, b) = (project(strand[k - 1]), project(strand[k]));
                    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs())).ceil().max(1.0) as usize;
                    for t in 0..=steps {
                        let f = t as f32 / steps as f32;
                        put(a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, [14, 9, 5], true);
                    }
                }
            }
        }
        let image = humanart_io::Image::new(
            humanart_io::Format::Png,
            w as u32,
            h as u32,
            3,
            humanart_io::ColorSpace::Srgb,
            humanart_io::Samples::U8(rgb),
        )
        .unwrap();
        let png = humanart_io::png::encode(&image).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/hair").join(name);
        // `name` may lead into a sibling folder (`../hair-render/...`): create the file's own.
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, png).unwrap();
    }
}
