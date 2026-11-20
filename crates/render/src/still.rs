//! High-quality stills at any resolution through tiled rendering `[QUAL-04]` `[QUAL-01]`.
//!
//! The image is cut into tiles no larger than the renderer's target; each tile is rendered with
//! the sub-frustum of its rectangle grown by an overlap on every side (so passes that read
//! neighbours, today the TAA neighbourhood and later the subsurface blur and bloom, see real data
//! at the tile edges), `samples` times with the same full-image jitter sequence, accumulated to
//! their exact mean, tone mapped, and only the tile's interior is kept. Shadow maps are fitted to
//! the scene, not the view, and per-pixel noise is seeded with the full-image pixel position, so a
//! tiled still equals the untiled render up to floating-point rounding (`AC-WP11-3`).

use humanart_math::{Vec2, halton};

use crate::camera::TileRect;
use crate::error::RenderError;

/// Largest still side.
pub const MAX_STILL_SIDE: u32 = 32_768;
/// Most samples per pixel.
pub const MAX_SAMPLES: u32 = 1024;

/// How a still is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillSettings {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Interior side of a tile (the target is this plus twice the overlap).
    pub tile: u32,
    /// Pixels added on every side of a tile and thrown away.
    pub overlap: u32,
    /// Jittered samples accumulated per pixel.
    pub samples: u32,
}

impl StillSettings {
    /// Checks the settings against `max_target`, the largest render target side.
    pub fn validate(&self, max_target: u32) -> Result<(), RenderError> {
        if self.width == 0 || self.height == 0 || self.width > MAX_STILL_SIDE || self.height > MAX_STILL_SIDE {
            return Err(RenderError::TooLarge {
                what: "still side",
                value: u64::from(self.width.max(self.height)),
                limit: u64::from(MAX_STILL_SIDE),
            });
        }
        if self.samples == 0 || self.samples > MAX_SAMPLES {
            return Err(RenderError::InvalidParameter { what: "samples per pixel outside 1..1024" });
        }
        let target = u64::from(self.tile) + 2 * u64::from(self.overlap);
        if self.tile == 0 || target > u64::from(max_target) {
            return Err(RenderError::TooLarge {
                what: "tile plus overlap",
                value: target,
                limit: u64::from(max_target),
            });
        }
        Ok(())
    }

    /// The target size each tile is rendered at.
    pub fn target_size(&self) -> (u32, u32) {
        let side = self.tile + 2 * self.overlap;
        // An image smaller than one tile renders at its own size plus the overlap.
        (side.min(self.width + 2 * self.overlap), side.min(self.height + 2 * self.overlap))
    }
}

/// One tile: the rectangle rendered (with overlap) and the interior kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlan {
    /// What is rendered, in full-image pixels (may reach outside the image).
    pub render: TileRect,
    /// What is kept, in full-image pixels (inside the image).
    pub keep: TileRect,
}

/// Cuts the image into tiles, row by row. The rendered rectangles all have the target size of
/// the settings.
pub fn plan_tiles(settings: &StillSettings) -> Vec<TilePlan> {
    let (tw, th) = settings.target_size();
    let interior_w = tw - 2 * settings.overlap;
    let interior_h = th - 2 * settings.overlap;
    let mut plans = Vec::new();
    let mut y = 0;
    while y < settings.height {
        let keep_h = interior_h.min(settings.height - y);
        let mut x = 0;
        while x < settings.width {
            let keep_w = interior_w.min(settings.width - x);
            plans.push(TilePlan {
                render: TileRect {
                    x: x as i32 - settings.overlap as i32,
                    y: y as i32 - settings.overlap as i32,
                    width: tw,
                    height: th,
                },
                keep: TileRect { x: x as i32, y: y as i32, width: keep_w, height: keep_h },
            });
            x += interior_w.max(1);
        }
        y += interior_h.max(1);
    }
    plans
}

/// The jitter of sample `i` (pixels, Halton 2/3 centred): the same for every tile and for the
/// untiled render.
pub fn sample_jitter(i: u32, samples: u32) -> Vec2 {
    if samples <= 1 {
        return Vec2::ZERO;
    }
    Vec2::new(halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5)
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::render_still;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use humanart_gpu::resources::Resources;

    use super::{StillSettings, plan_tiles, sample_jitter};
    use crate::camera::Camera;
    use crate::capture::read_texture;
    use crate::error::RenderError;
    use crate::light::LightRig;
    use crate::renderer::{FrameInput, Renderer, StillFrame};
    use crate::scene::GpuScene;

    /// Renders a still with `renderer`, which must be configured at `settings.target_size()`
    /// (with TAA on, so frames accumulate). Returns tightly packed sRGB-encoded RGBA8 rows of the
    /// full image. Blocks on the GPU (stills, the command-line tool).
    pub fn render_still(
        renderer: &mut Renderer,
        resources: &mut Resources,
        scene: &mut GpuScene,
        camera: &Camera,
        rig: &LightRig,
        settings: &StillSettings,
    ) -> Result<Vec<u8>, RenderError> {
        let (tw, th) = settings.target_size();
        let config = *renderer.config();
        settings.validate(config.width.max(config.height))?;
        if config.width != tw || config.height != th || !config.taa {
            return Err(RenderError::InvalidParameter {
                what: "configure the renderer at the still's tile size with TAA on",
            });
        }
        let row = settings.width as usize * 4;
        let mut image = vec![0_u8; row * settings.height as usize];
        let device = resources.device().clone();
        for plan in plan_tiles(settings) {
            for sample in 0..settings.samples {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("still tile") });
                renderer.render(
                    resources,
                    &mut encoder,
                    FrameInput {
                        camera,
                        rig,
                        scene,
                        overlay: &[],
                        wireframe: false,
                        still: Some(StillFrame {
                            full_width: settings.width,
                            full_height: settings.height,
                            tile: plan.render,
                            jitter: sample_jitter(sample, settings.samples),
                            sample,
                        }),
                        pick: None,
                        reset_history: sample == 0,
                    },
                )?;
                resources.queue().submit([encoder.finish()]);
                renderer.after_submit();
            }
            let display =
                renderer.display_texture().ok_or(RenderError::InvalidParameter { what: "no display target" })?;
            let (w, _, pixels) = read_texture(resources, display, 0)?;
            let src_row = w as usize * 4;
            let ox = (plan.keep.x - plan.render.x) as usize;
            let oy = (plan.keep.y - plan.render.y) as usize;
            for y in 0..plan.keep.height as usize {
                let src = (oy + y) * src_row + ox * 4;
                let dst = (plan.keep.y as usize + y) * row + plan.keep.x as usize * 4;
                let len = plan.keep.width as usize * 4;
                let (Some(from), Some(to)) = (pixels.get(src..src + len), image.get_mut(dst..dst + len)) else {
                    return Err(RenderError::InvalidParameter { what: "a tile outside the image" });
                };
                to.copy_from_slice(from);
            }
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qual_04_tiles_cover_the_image_exactly_once() {
        for (w, h, tile, overlap) in [(7680, 4320, 2048, 32), (100, 50, 64, 8), (64, 64, 64, 0), (1000, 1, 300, 16)] {
            let s = StillSettings { width: w, height: h, tile, overlap, samples: 4 };
            s.validate(8192).unwrap();
            let mut covered = vec![0_u8; (w * h) as usize];
            let (tw, th) = s.target_size();
            for p in plan_tiles(&s) {
                assert_eq!((p.render.width, p.render.height), (tw, th));
                assert!(p.keep.x - p.render.x >= overlap as i32 || overlap == 0);
                assert!(p.keep.x + p.keep.width as i32 <= p.render.x + p.render.width as i32);
                for y in p.keep.y..p.keep.y + p.keep.height as i32 {
                    for x in p.keep.x..p.keep.x + p.keep.width as i32 {
                        covered[(y as u32 * w + x as u32) as usize] += 1;
                    }
                }
            }
            assert!(covered.iter().all(|&c| c == 1), "{w}x{h}");
        }
    }

    #[test]
    fn settings_are_checked() {
        let s = StillSettings { width: 7680, height: 4320, tile: 8192, overlap: 32, samples: 4 };
        assert!(s.validate(8192).is_err(), "tile plus overlap above the target limit");
        assert!(StillSettings { samples: 0, ..s }.validate(8192).is_err());
        assert!(StillSettings { width: 0, ..s }.validate(8192).is_err());
    }

    #[test]
    fn jitter_is_centred_and_shared() {
        assert_eq!(sample_jitter(0, 1), Vec2::ZERO);
        let mean = (0..64).map(|i| sample_jitter(i, 64)).fold(Vec2::ZERO, |a, b| a + b) / 64.0;
        assert!(mean.length() < 0.02, "{mean:?}");
        assert!((0..64).all(|i| sample_jitter(i, 64).x.abs() <= 0.5));
    }
}
