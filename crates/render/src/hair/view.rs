//! The constant blocks of the hair passes and the per-view decisions behind them: the deep opacity
//! map frames of the lights `[HR-03]`, the level of detail of a groom (strand thinning with width
//! compensation, segment joining, `[HR-04]`) and the dispatch grid of the compute rasteriser
//! `[HR-01]`.
//!
//! [`ViewBlock`] mirrors `HairView` and [`DrawBlock`] `HairDraw` of `shaders/hair/common.wgsl`.

use humanart_gpu::pod_struct;
use humanart_math::Vec3;

use super::HairError;
use crate::light::{Light, LightKind};

/// Most grooms drawn in one view (`HAIR_MAX_GROOMS`).
pub const MAX_GROOMS: usize = 16;
/// Most lights with a deep opacity map (`HAIR_MAX_DOMS`).
pub const MAX_DOMS: usize = 2;
/// Fixed-point unit of the per-pixel coverage sum (`HAIR_COVERAGE_ONE`).
pub const COVERAGE_ONE: f32 = 256.0;
/// Where the per-pixel coverage sum stops growing, in coverage units: alpha is then
/// `1 - exp(-16)`.
pub const COVERAGE_SATURATION: f32 = 16.0;
/// Fixed-point unit of the voxel fibre density, square metres (`HAIR_VOXEL_UNIT`).
pub const VOXEL_UNIT: f32 = 1e-9;
/// Mean projected area of a thin cylinder over random orientations per unit of side area
/// (`pi / 4`): side area per volume times this is the optical depth per metre.
pub const VOXEL_TAU_FACTOR: f32 = std::f32::consts::FRAC_PI_4;
/// Far edges of the four deep-opacity layers behind the first fibre, metres (the last one is
/// where the open-ended layer stops growing).
pub const DOM_LAYERS: [f32; 4] = [0.003, 0.009, 0.02, 0.06];
/// Flag: dual scattering (`HAIR_FLAG_DUAL`).
pub const FLAG_DUAL: u32 = 1;
/// Flag: density-grid ambient occlusion (`HAIR_FLAG_OCCLUSION`).
pub const FLAG_OCCLUSION: u32 = 2;
/// Flag: hair shadows on the skin (`HAIR_FLAG_SKIN_SHADOW`).
pub const FLAG_SKIN_SHADOW: u32 = 4;
/// Flag: skip segments hidden behind the opaque scene (`HAIR_FLAG_CULL`).
pub const FLAG_CULL: u32 = 8;
/// Side of the scene depth tiles of the hidden-segment test, pixels (`HAIR_TILE`).
pub const TILE: u32 = 16;
/// Invocations per workgroup of the compute rasteriser.
pub const WORKGROUP: u32 = 64;
/// Most workgroups along one dispatch axis.
pub const MAX_GROUPS: u32 = 65_535;

/// Projected strand width (pixels) below which the strands are thinned out `[HR-04]`.
pub const LOD_WIDTH_PX: f32 = 0.3;
/// The smallest fraction of strands the far field keeps.
pub const LOD_MIN_KEEP: f32 = 1.0 / 32.0;
/// The shortest projected segment (pixels) before neighbouring segments are joined.
pub const LOD_SEGMENT_PX: f32 = 2.0;
/// The longest walk (pixels along the major axis) one raster thread is given: longer projected
/// segments are split over [`GroomLod::split`] threads, each walking its part.
pub const LOD_WALK_PX: f32 = 8.0;
/// The most threads one segment's walk is split over.
pub const LOD_MAX_SPLIT: u32 = 8;
/// Most strands drawn into the deep opacity maps (more are subsampled with coverage compensation).
pub const DOM_STRANDS: u32 = 32_768;

pod_struct! {
    /// The per-view constants of the hair passes (`struct HairView`).
    #[derive(Debug, PartialEq)]
    pub struct ViewBlock(size = 736, align = 4) {
        /// Target width, height, focal length in pixels, coverage saturation (fixed point).
        pub screen: [f32; 4],
        /// Per deep opacity map: right axis, centre x.
        pub dom_right: [[f32; 4]; MAX_DOMS],
        /// Per map: up axis, centre y.
        pub dom_up: [[f32; 4]; MAX_DOMS],
        /// Per map: the light's travel direction, centre z.
        pub dom_forward: [[f32; 4]; MAX_DOMS],
        /// Per map: radius, light slot, texel size (metres), self-shadow bias (metres).
        pub dom_params: [[f32; 4]; MAX_DOMS],
        /// Far edges of the opacity layers, metres.
        pub dom_layers: [f32; 4],
        /// Voxel grid origin and cell size, metres.
        pub voxel: [f32; 4],
        /// Longitudinal variances v0..v3.
        pub variances: [f32; 4],
        /// Lobe shifts R, TT, TRT; voxel area density to optical depth factor.
        pub shifts: [f32; 4],
        /// Voxels per side, deep opacity maps, grooms, flags.
        pub counts: [u32; 4],
        /// Per groom: natural sigma_a and variation; dyed sigma_a and dye amount.
        pub grooms: [[f32; 8]; MAX_GROOMS],
    }
}

pod_struct! {
    /// The per-groom constants of a hair draw (`struct HairDraw`).
    #[derive(Debug, PartialEq)]
    pub struct DrawBlock(size = 48, align = 4) {
        /// Strands, points per strand, point stride, groom index.
        pub counts: [u32; 4],
        /// Segments per strand after the stride, threads per dispatch row, keep threshold, deep
        /// opacity strand step.
        pub dispatch: [u32; 4],
        /// Coverage scale, deep-opacity coverage scale, width scale, walk parts per segment
        /// ([`GroomLod::split`], 1 for the ribbon path).
        pub scale: [f32; 4],
    }
}

impl ViewBlock {
    /// Every field zero.
    pub const ZERO: Self = Self {
        screen: [0.0; 4],
        dom_right: [[0.0; 4]; MAX_DOMS],
        dom_up: [[0.0; 4]; MAX_DOMS],
        dom_forward: [[0.0; 4]; MAX_DOMS],
        dom_params: [[0.0; 4]; MAX_DOMS],
        dom_layers: [0.0; 4],
        voxel: [0.0; 4],
        variances: [0.0; 4],
        shifts: [0.0; 4],
        counts: [0; 4],
        grooms: [[0.0; 8]; MAX_GROOMS],
    };
}

/// The orthographic frame of one light's deep opacity map around the hair `[HR-03]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomFrame {
    /// Across the map.
    pub right: Vec3,
    /// Up the map.
    pub up: Vec3,
    /// The light's travel direction (into the map).
    pub forward: Vec3,
    /// The centre of the hair's bounding sphere.
    pub centre: Vec3,
    /// Half the map's side and depth, metres.
    pub radius: f32,
}

impl DomFrame {
    /// The frame of `light` around the sphere (`centre`, `radius`): along a directional light's
    /// travel, or from a spot or rect light's position towards the centre. `None` when the
    /// sphere is degenerate or the light sits at its centre.
    pub fn fit(light: &Light, centre: Vec3, radius: f32) -> Option<Self> {
        if !(radius > 0.0 && radius.is_finite()) {
            return None;
        }
        let forward = match light.kind {
            LightKind::Directional { .. } => light.direction,
            LightKind::Spot { .. } | LightKind::Rect { .. } => centre - light.position,
        };
        let length = forward.length();
        if !(length > 1e-6 && length.is_finite()) {
            return None;
        }
        let forward = forward * (1.0 / length);
        let helper = if forward.y.abs() < 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let right = forward.cross(helper).normalize_or(Vec3::new(1.0, 0.0, 0.0));
        let up = right.cross(forward);
        Some(Self { right, up, forward, centre, radius })
    }

    /// `p` in map space: x, y in [-1, 1] across the map, z the reversed depth in [0, 1] (1 nearest
    /// the light), as `hair_dom_local` computes it.
    pub fn local(&self, p: Vec3) -> Vec3 {
        let d = p - self.centre;
        Vec3::new(
            d.dot(self.right) / self.radius,
            d.dot(self.up) / self.radius,
            0.5 - d.dot(self.forward) / (2.0 * self.radius),
        )
    }
}

/// How much `light` illuminates a point at `p` (its illuminance there, lux), for choosing which
/// lights get a deep opacity map.
pub fn light_weight(light: &Light, p: Vec3) -> f32 {
    let w = light.illuminance_at(p);
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// The lights that get a deep opacity map: the [`MAX_DOMS`] that light `p` most, as indices into
/// `lights` (slots of the frame's light array), strongest first.
pub fn dom_lights(lights: &[Light], p: Vec3) -> ([usize; MAX_DOMS], usize) {
    let mut best = [(0.0_f32, 0_usize); MAX_DOMS];
    let mut count = 0;
    for (i, light) in lights.iter().enumerate().take(crate::light::MAX_LIGHTS) {
        let w = light_weight(light, p);
        if w <= 0.0 {
            continue;
        }
        // Insertion into the sorted short list.
        let mut at = count.min(MAX_DOMS);
        while at > 0 && best[at - 1].0 < w {
            at -= 1;
        }
        if at < MAX_DOMS {
            let mut k = MAX_DOMS - 1;
            while k > at {
                best[k] = best[k - 1];
                k -= 1;
            }
            best[at] = (w, i);
            count = (count + 1).min(MAX_DOMS);
        }
    }
    (best.map(|(_, i)| i), count)
}

/// What the level of detail of one groom sees `[HR-04]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodInput {
    /// Strands of the groom.
    pub strands: u32,
    /// Points per strand.
    pub points: u32,
    /// Mean strand diameter, metres.
    pub diameter: f32,
    /// Mean strand length, metres.
    pub length: f32,
    /// Distance from the eye to the nearest point of the groom's bounds, metres.
    pub distance: f32,
    /// Focal length in pixels (pixels per metre at one metre).
    pub focal_px: f32,
    /// The least fraction of the strands the thinning keeps, at least [`LOD_MIN_KEEP`] (1 never
    /// thins: sparse grooms such as peach fuzz, brows and lashes, whose few survivors would show
    /// as isolated specks with the scaled-up coverage).
    pub min_keep: f32,
}

/// The level of detail of one groom in one view `[HR-04]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroomLod {
    /// Fraction of the strands drawn.
    pub keep: f32,
    /// Hash threshold of the kept strands (`hair_kept`).
    pub keep_threshold: u32,
    /// Coverage of each kept strand per unit of its own (the width compensation, `1 / keep`).
    pub coverage_scale: f32,
    /// Points joined into one drawn segment.
    pub stride: u32,
    /// Drawn segments per strand.
    pub segments: u32,
    /// Strand step of the deep opacity maps.
    pub dom_step: u32,
    /// Raster threads per drawn segment (1..=[`LOD_MAX_SPLIT`]): the projected segment length
    /// over [`LOD_WALK_PX`], so a close-up's long walks do not hold a whole wave of threads on
    /// one lane.
    pub split: u32,
}

impl GroomLod {
    /// Chooses the level of detail: strands thinner than [`LOD_WIDTH_PX`] are thinned out down to
    /// [`LOD_MIN_KEEP`] with the survivors' coverage scaled up so the hair keeps its density
    /// (uniform hash pruning keeps every clump populated in proportion), and segments shorter than
    /// [`LOD_SEGMENT_PX`] are joined.
    pub fn select(input: &LodInput) -> Result<Self, HairError> {
        if input.points < 2 {
            return Err(HairError::Invalid { what: "a groom needs at least two points per strand" });
        }
        let distance = if input.distance.is_finite() { input.distance.max(1e-3) } else { 1e-3 };
        let focal = if input.focal_px.is_finite() { input.focal_px.max(0.0) } else { 0.0 };
        let width_px = input.diameter.max(0.0) * focal / distance;
        let floor = if input.min_keep.is_finite() { input.min_keep.clamp(LOD_MIN_KEEP, 1.0) } else { LOD_MIN_KEEP };
        let keep =
            if width_px.is_finite() && width_px < LOD_WIDTH_PX { (width_px / LOD_WIDTH_PX).max(floor) } else { 1.0 };
        let max_segments = input.points - 1;
        let segment_px = input.length.max(0.0) / max_segments as f32 * focal / distance;
        let stride = if segment_px.is_finite() && segment_px > 0.0 && segment_px < LOD_SEGMENT_PX {
            ((LOD_SEGMENT_PX / segment_px).ceil() as u32).clamp(1, max_segments)
        } else if segment_px.is_finite() && segment_px > 0.0 {
            1
        } else {
            max_segments
        };
        let segments = max_segments.div_ceil(stride);
        let drawn_px = segment_px * stride as f32;
        let split = if drawn_px.is_finite() && drawn_px > LOD_WALK_PX {
            ((drawn_px / LOD_WALK_PX).ceil() as u32).clamp(1, LOD_MAX_SPLIT)
        } else {
            1
        };
        let keep_threshold = if keep >= 1.0 { u32::MAX } else { (f64::from(keep) * f64::from(u32::MAX)) as u32 };
        let dom_step = input.strands.div_ceil(DOM_STRANDS).max(1);
        Ok(Self { keep, keep_threshold, coverage_scale: 1.0 / keep, stride, segments, dom_step, split })
    }
}

/// A dispatch of one thread per drawn segment: workgroups along x and y, and threads per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// Workgroups along x.
    pub x: u32,
    /// Workgroups along y.
    pub y: u32,
    /// Threads per row (`x * WORKGROUP`).
    pub row: u32,
}

impl Grid {
    /// The grid for `strands` strands of `segments` drawn segments each.
    pub fn new(strands: u32, segments: u32) -> Result<Self, HairError> {
        let threads = u64::from(strands) * u64::from(segments);
        let threads = u32::try_from(threads).map_err(|_| HairError::TooLarge {
            what: "hair segments in one dispatch",
            value: threads,
            limit: u64::from(u32::MAX),
        })?;
        let groups = threads.div_ceil(WORKGROUP);
        if groups == 0 {
            return Ok(Self { x: 0, y: 0, row: 0 });
        }
        let x = groups.min(MAX_GROUPS);
        let y = groups.div_ceil(x);
        if y > MAX_GROUPS {
            return Err(HairError::TooLarge {
                what: "hair dispatch rows",
                value: u64::from(y),
                limit: u64::from(MAX_GROUPS),
            });
        }
        Ok(Self { x, y, row: x * WORKGROUP })
    }
}

/// The PCG hash of `hair_hash` (mirrors `humanart_model::hair::hash_u32`).
pub const fn hair_hash(x: u32) -> u32 {
    humanart_model::hair::hash_u32(x)
}

/// Whether strand `strand` survives the pruning threshold `keep` (`hair_kept`).
pub const fn hair_kept(strand: u32, keep: u32) -> bool {
    hair_hash(strand ^ 0x5bd1_e995) <= keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::Light;

    #[test]
    fn hr_03_dom_frame_puts_the_light_side_nearest() {
        let light = Light::directional(Vec3::new(0.3, -1.0, -0.2), 10_000.0, 5600.0);
        let centre = Vec3::new(0.0, 1.6, 0.0);
        let f = DomFrame::fit(&light, centre, 0.2).unwrap();
        assert!((f.right.dot(f.up)).abs() < 1e-5 && (f.right.dot(f.forward)).abs() < 1e-5);
        assert!((f.up.length() - 1.0).abs() < 1e-5);
        // The point towards the light is nearer (larger reversed depth) than the point away.
        let toward = f.local(centre - f.forward * 0.1);
        let away = f.local(centre + f.forward * 0.1);
        assert!(toward.z > away.z && (0.0..=1.0).contains(&toward.z) && (0.0..=1.0).contains(&away.z));
        assert!(f.local(centre).length() > 0.0 && (f.local(centre).z - 0.5).abs() < 1e-6);
        // A spot light aims from its position at the centre.
        let spot = Light::spot(Vec3::new(1.0, 1.6, 0.0), centre, 100.0, 0.5, 5600.0);
        let s = DomFrame::fit(&spot, centre, 0.2).unwrap();
        assert!((s.forward - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        assert!(DomFrame::fit(&light, centre, 0.0).is_none());
    }

    #[test]
    fn hr_03_dom_lights_are_the_strongest_two() {
        let p = Vec3::new(0.0, 1.6, 0.0);
        let lights = [
            Light::directional(Vec3::new(0.0, -1.0, 0.0), 100.0, 5600.0),
            Light::directional(Vec3::new(1.0, 0.0, 0.0), 5000.0, 5600.0),
            Light::directional(Vec3::new(0.0, 0.0, -1.0), 800.0, 5600.0),
        ];
        let (order, n) = dom_lights(&lights, p);
        assert_eq!((order, n), ([1, 2], 2));
        let (order, n) = dom_lights(&lights[..1], p);
        assert_eq!((order[0], n), (0, 1));
        assert_eq!(dom_lights(&[], p).1, 0);
    }

    #[test]
    fn hr_04_lod_thins_far_hair_and_keeps_its_density() {
        let near = LodInput {
            strands: 100_000,
            points: 24,
            diameter: 7e-5,
            length: 0.3,
            distance: 0.4,
            focal_px: 4000.0,
            min_keep: 0.0,
        };
        let lod = GroomLod::select(&near).unwrap();
        assert_eq!((lod.keep, lod.keep_threshold, lod.stride), (1.0, u32::MAX, 1));
        assert_eq!(lod.segments, 23);
        assert_eq!(lod.dom_step, 4);
        // 130 px segments: their walks split over the most threads.
        assert_eq!(lod.split, LOD_MAX_SPLIT);
        // Ten times further: 0.07 px wide strands, about a quarter kept at four times the coverage.
        let far = GroomLod::select(&LodInput { distance: 4.0, ..near }).unwrap();
        assert!(far.keep < 0.3 && far.keep > 0.2, "{far:?}");
        assert!((far.keep * far.coverage_scale - 1.0).abs() < 1e-6);
        // 13 px segments stay; at 40 m they are 1.3 px and pairs are joined.
        assert_eq!((far.stride, far.split), (1, 2));
        let farther = GroomLod::select(&LodInput { distance: 40.0, ..near }).unwrap();
        assert_eq!((farther.stride, farther.segments, farther.split), (2, 12, 1));
        // The pruning hash keeps about the asked fraction of strands.
        let kept = (0..100_000_u32).filter(|&s| hair_kept(s, far.keep_threshold)).count() as f32 / 1e5;
        assert!((kept - far.keep).abs() < 0.01, "kept {kept} for {}", far.keep);
        // Very far: the floor.
        let tiny = GroomLod::select(&LodInput { distance: 1000.0, ..near }).unwrap();
        // A sparse groom is never thinned below its floor.
        let sparse = GroomLod::select(&LodInput { distance: 1000.0, min_keep: 1.0, ..near }).unwrap();
        assert!((sparse.keep - 1.0).abs() < 1e-6 && sparse.keep_threshold == u32::MAX, "{sparse:?}");
        assert!((sparse.coverage_scale - 1.0).abs() < 1e-6);
        assert_eq!(tiny.keep, LOD_MIN_KEEP);
        assert_eq!(tiny.segments, 1);
        assert!(GroomLod::select(&LodInput { points: 1, ..near }).is_err());
        // Hostile numbers stay finite.
        let nan = GroomLod::select(&LodInput { distance: f32::NAN, focal_px: f32::INFINITY, ..near }).unwrap();
        assert!(nan.keep.is_finite() && nan.segments >= 1 && (1..=LOD_MAX_SPLIT).contains(&nan.split));
    }

    #[test]
    fn hr_01_dispatch_grid_covers_every_segment() {
        let g = Grid::new(100_000, 23).unwrap();
        assert!(u64::from(g.x) * u64::from(g.y) * u64::from(WORKGROUP) >= 2_300_000);
        assert_eq!(g.row, g.x * WORKGROUP);
        let big = Grid::new(10_000_000, 31).unwrap();
        assert!(big.y > 1 && big.x == MAX_GROUPS);
        assert_eq!(Grid::new(0, 5).unwrap(), Grid { x: 0, y: 0, row: 0 });
        assert!(Grid::new(u32::MAX, 3).is_err());
    }
}
