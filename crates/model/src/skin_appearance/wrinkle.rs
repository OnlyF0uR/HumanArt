//! Wrinkle and tension map blending `[SKN-30]`, driven by the edge strain of `[EVAL-06]` through
//! the wrinkle drivers of `[EXP-04]` ([`crate::eval::wrinkle`]). This is the CPU reference of
//! `skin/wrinkle.wgsl` and `skin/wrinkle_drivers.wgsl`.
//!
//! **Maps.** A [`WrinkleMap`] belongs to one driver and covers a rectangle of one UV tile: a height
//! delta in mm at full fold (negative in a crease), a cavity delta at full fold, and a `flatten`
//! factor, the share of the micro-detail contrast that a fully stretched region loses. Maps are
//! authored by the generators (a `features::GeneratorKind::Wrinkles` content layer converts with
//! [`WrinkleMap::from_content`]) or imported.
//!
//! **Baked layout.** [`WrinkleMaps::bake`] merges the maps into per-texel **slots**: every texel holds
//! up to [`WRINKLE_SLOTS`] drivers (forehead rows and frown lines overlap), each with its height,
//! height slope along U and V (central differences of that driver's own height, so the shading pass
//! perturbs the normal without sampling neighbours) and cavity. Maps of the same driver add into one
//! slot (one activation scales them all, so the sum is exact); where more than two drivers meet,
//! the two with the largest height magnitude are kept. On the GPU a tile is three `rgba16float`
//! layers: `(h0, su0, sv0, c0)`, `(h1, su1, sv1, c1)`, `(id0, id1, flatten0, flatten1)` with
//! `id = driver + 1` (0 = empty; f16 holds integers exactly up to 2048, hence
//! [`MAX_WRINKLE_DRIVERS`]).
//!
//! **Blending** at a texel with fold activations `f` and stretch activations `t` (one per driver,
//! [`DriverTable::activate`]):
//!
//! - height `= Σ_k f[id_k] · h_k` (compression deepens folds), slope `= Σ_k f[id_k] · s_k`,
//!   cavity delta `= Σ_k f[id_k] · c_k`;
//! - micro-detail scale `= clamp(1 − Σ_k t[id_k] · flatten_k, 0, 1)` (stretch reduces micro-detail
//!   contrast).
//!
//! **Activations.** [`DriverTable`] holds the drivers' gains in flat arrays. Per frame the CPU sums
//! the expression part ([`DriverTable::expression_terms`], allocation-free); the GPU adds the
//! measured region strain, which never leaves the GPU: `fold = clamp(term + g_c · compression_r, 0, 1)`
//! and `stretch = clamp(g_s · stretch_r, 0, 1)`, exactly [`crate::eval::WrinkleDrivers::evaluate`].

use super::dirty::{MAX_TILES, Rect};
use super::error::SkinError;
use super::layers::{LayerContent, MAX_CONTENT_SIDE};
use crate::eval::wrinkle::{MAX_DRIVER_EXPRESSIONS, MAX_GAIN, WrinkleDriver};

/// Drivers per texel.
pub const WRINKLE_SLOTS: usize = 2;
/// Most drivers the blending addresses (driver ids are stored as f16 integers).
pub const MAX_WRINKLE_DRIVERS: usize = 2048;
/// Largest wrinkle-map tile edge in texels (a baked tile holds 48 bytes per texel on the CPU).
pub const MAX_WRINKLE_SIZE: u32 = 2048;
/// Most UV tiles that carry wrinkle maps.
pub const MAX_WRINKLE_TILES: usize = 8;
/// Most maps in one bake.
pub const MAX_WRINKLE_MAPS: usize = 4096;
/// `rgba16float` layers per tile on the GPU.
pub const WRINKLE_LAYERS: usize = 3;

/// One driver's wrinkle map over a rectangle of one UV tile.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleMap {
    /// The driver (index into the [`DriverTable`]).
    pub driver: u16,
    /// The UV tile.
    pub tile: u32,
    /// Where the texels lie, in wrinkle-map texels of the tile.
    pub rect: Rect,
    /// Height delta in mm at full fold, row-major in `rect`.
    pub height: Vec<f32>,
    /// Cavity delta at full fold (negative darkens the crease), row-major in `rect`.
    pub cavity: Vec<f32>,
    /// Share of the micro-detail contrast lost at full stretch, 0..1.
    pub flatten: f32,
}

impl WrinkleMap {
    /// A map from a content layer whose first two components are height and cavity (the wrinkle
    /// and lip-line generators' layout), weighted by its coverage.
    pub fn from_content(driver: u16, content: &LayerContent, flatten: f32) -> Result<Self, SkinError> {
        let n = content.values.len();
        if content.coverage.len() != n || n != content.rect.w as usize * content.rect.h as usize {
            return Err(SkinError::Invalid { what: "content values and coverage must fill the rectangle" });
        }
        let height = content.values.iter().zip(&content.coverage).map(|(v, c)| v[0] * c).collect();
        let cavity = content.values.iter().zip(&content.coverage).map(|(v, c)| v[1] * c).collect();
        let map = Self { driver, tile: content.tile, rect: content.rect, height, cavity, flatten };
        map.check()?;
        Ok(map)
    }

    fn check(&self) -> Result<(), SkinError> {
        let n = self.rect.w as usize * self.rect.h as usize;
        if self.rect.w == 0 || self.rect.h == 0 || self.rect.w > MAX_CONTENT_SIDE || self.rect.h > MAX_CONTENT_SIDE {
            return Err(SkinError::Invalid { what: "a wrinkle map needs a non-empty rectangle" });
        }
        if self.height.len() != n || self.cavity.len() != n {
            return Err(SkinError::Invalid { what: "wrinkle map arrays must fill the rectangle" });
        }
        if !(self.flatten.is_finite() && (0.0..=1.0).contains(&self.flatten)) {
            return Err(SkinError::Invalid { what: "wrinkle flatten must be in 0..1" });
        }
        if !self.height.iter().chain(&self.cavity).all(|v| v.is_finite()) {
            return Err(SkinError::Invalid { what: "wrinkle map values must be finite" });
        }
        Ok(())
    }

    /// The height at rect-local `(x, y)`, 0 outside the rectangle.
    fn height_at(&self, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= i64::from(self.rect.w) || y >= i64::from(self.rect.h) {
            return 0.0;
        }
        self.height[y as usize * self.rect.w as usize + x as usize]
    }
}

/// The baked state of one texel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WrinkleTexel {
    /// Driver + 1 per slot (0 = empty).
    pub id: [u16; WRINKLE_SLOTS],
    /// Height delta in mm at full fold.
    pub height: [f32; WRINKLE_SLOTS],
    /// Height slope (mm per mm) along +U and +V at full fold.
    pub slope: [[f32; 2]; WRINKLE_SLOTS],
    /// Cavity delta at full fold.
    pub cavity: [f32; WRINKLE_SLOTS],
    /// Micro-detail contrast lost at full stretch.
    pub flatten: [f32; WRINKLE_SLOTS],
}

/// The blended wrinkle state at one texel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WrinkleSample {
    /// Height delta in mm.
    pub height: f32,
    /// Height slope along +U and +V (mm per mm).
    pub slope: [f32; 2],
    /// Cavity delta.
    pub cavity: f32,
    /// Factor on the micro-detail contrast, 0..1.
    pub detail_scale: f32,
}

/// Blends one baked texel with the drivers' fold and stretch activations (missing or non-finite
/// activations count as 0).
pub fn blend_texel(t: &WrinkleTexel, fold: &[f32], stretch: &[f32]) -> WrinkleSample {
    let read = |a: &[f32], id: u16| {
        id.checked_sub(1).and_then(|d| a.get(d as usize)).copied().filter(|v| v.is_finite()).unwrap_or(0.0)
    };
    let mut out = WrinkleSample::default();
    let mut lost = 0.0;
    for k in 0..WRINKLE_SLOTS {
        let f = read(fold, t.id[k]);
        out.height += f * t.height[k];
        out.slope[0] += f * t.slope[k][0];
        out.slope[1] += f * t.slope[k][1];
        out.cavity += f * t.cavity[k];
        lost += read(stretch, t.id[k]) * t.flatten[k];
    }
    out.detail_scale = (1.0 - lost).clamp(0.0, 1.0);
    out
}

/// The baked wrinkle maps of the UV tiles that have any.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleMaps {
    size: u32,
    tiles: Vec<u32>,
    texels: Vec<WrinkleTexel>,
}

impl WrinkleMaps {
    /// Bakes `maps` into `size²`-texel tiles; `texel_mm[k]` is the edge of one wrinkle-map texel in
    /// mm on UV tile `tiles[k]` (for the slopes).
    pub fn bake(size: u32, tiles: &[u32], texel_mm: &[f32], maps: &[WrinkleMap]) -> Result<Self, SkinError> {
        if size == 0 || size > MAX_WRINKLE_SIZE {
            return Err(SkinError::limit("wrinkle map size", u64::from(MAX_WRINKLE_SIZE), u64::from(size)));
        }
        if tiles.len() > MAX_WRINKLE_TILES {
            return Err(SkinError::limit("wrinkle tiles", MAX_WRINKLE_TILES as u64, tiles.len() as u64));
        }
        if maps.len() > MAX_WRINKLE_MAPS {
            return Err(SkinError::limit("wrinkle maps", MAX_WRINKLE_MAPS as u64, maps.len() as u64));
        }
        if texel_mm.len() != tiles.len() || !texel_mm.iter().all(|t| t.is_finite() && *t > 0.0) {
            return Err(SkinError::Invalid { what: "every wrinkle tile needs a positive texel size" });
        }
        if tiles.iter().any(|&t| t >= MAX_TILES) || (1..tiles.len()).any(|i| tiles[..i].contains(&tiles[i])) {
            return Err(SkinError::Invalid { what: "wrinkle tiles must be distinct tile indices" });
        }
        let per_tile = size as usize * size as usize;
        let mut out =
            Self { size, tiles: tiles.to_vec(), texels: vec![WrinkleTexel::default(); per_tile * tiles.len()] };
        for map in maps {
            map.check()?;
            if usize::from(map.driver) >= MAX_WRINKLE_DRIVERS {
                return Err(SkinError::limit("wrinkle driver", MAX_WRINKLE_DRIVERS as u64, u64::from(map.driver)));
            }
            let Some(slot) = tiles.iter().position(|&t| t == map.tile) else {
                return Err(SkinError::Invalid { what: "a wrinkle map lies on a tile without wrinkle maps" });
            };
            let r = map.rect;
            if u64::from(r.x) + u64::from(r.w) > u64::from(size) || u64::from(r.y) + u64::from(r.h) > u64::from(size) {
                return Err(SkinError::Invalid { what: "a wrinkle map rectangle leaves its tile" });
            }
            let inv = 1.0 / (2.0 * texel_mm[slot]);
            let id = map.driver + 1;
            for y in 0..r.h {
                for x in 0..r.w {
                    let (lx, ly) = (i64::from(x), i64::from(y));
                    let h = map.height_at(lx, ly);
                    let su = (map.height_at(lx + 1, ly) - map.height_at(lx - 1, ly)) * inv;
                    let sv = (map.height_at(lx, ly + 1) - map.height_at(lx, ly - 1)) * inv;
                    let c = map.cavity[(y * r.w + x) as usize];
                    let texel = &mut out.texels[slot * per_tile + ((r.y + y) * size + r.x + x) as usize];
                    insert(texel, id, h, [su, sv], c, map.flatten);
                }
            }
        }
        Ok(out)
    }

    /// Edge length in texels.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The UV tiles that carry maps, in GPU layer order (tile `tiles()[k]` is layers `3k..3k+3`).
    pub fn tiles(&self) -> &[u32] {
        &self.tiles
    }

    /// The baked texel `(x, y)` of UV tile `tile` (`None` for a tile without maps or outside).
    pub fn texel(&self, tile: u32, x: u32, y: u32) -> Option<&WrinkleTexel> {
        let slot = self.tiles.iter().position(|&t| t == tile)?;
        if x >= self.size || y >= self.size {
            return None;
        }
        let per_tile = self.size as usize * self.size as usize;
        self.texels.get(slot * per_tile + (y * self.size + x) as usize)
    }

    /// Writes GPU layer `layer` (0..3) of the `k`-th wrinkle tile as RGBA values, row-major, into
    /// `out` (`size²` entries).
    pub fn gpu_layer(&self, k: usize, layer: usize, out: &mut [[f32; 4]]) -> Result<(), SkinError> {
        let per_tile = self.size as usize * self.size as usize;
        if k >= self.tiles.len() || layer >= WRINKLE_LAYERS || out.len() != per_tile {
            return Err(SkinError::Invalid { what: "no such wrinkle layer, or the output has the wrong size" });
        }
        for (o, t) in out.iter_mut().zip(&self.texels[k * per_tile..(k + 1) * per_tile]) {
            *o = match layer {
                0 | 1 => [t.height[layer], t.slope[layer][0], t.slope[layer][1], t.cavity[layer]],
                _ => [f32::from(t.id[0]), f32::from(t.id[1]), t.flatten[0], t.flatten[1]],
            };
        }
        Ok(())
    }
}

/// Adds one driver's contribution to a texel's slots.
fn insert(t: &mut WrinkleTexel, id: u16, h: f32, s: [f32; 2], c: f32, flatten: f32) {
    if h == 0.0 && c == 0.0 && s == [0.0, 0.0] {
        return;
    }
    let k = if let Some(k) = t.id.iter().position(|&x| x == id) {
        k
    } else if let Some(k) = t.id.iter().position(|&x| x == 0) {
        t.id[k] = id;
        t.flatten[k] = flatten;
        k
    } else {
        // Both slots taken by other drivers: replace the weaker one if this one is stronger.
        let weak = usize::from(t.height[1].abs() < t.height[0].abs());
        if h.abs() <= t.height[weak].abs() {
            return;
        }
        t.id[weak] = id;
        t.flatten[weak] = flatten;
        t.height[weak] = 0.0;
        t.slope[weak] = [0.0, 0.0];
        t.cavity[weak] = 0.0;
        weak
    };
    t.height[k] += h;
    t.slope[k][0] += s[0];
    t.slope[k][1] += s[1];
    t.cavity[k] += c;
    t.flatten[k] = t.flatten[k].max(flatten);
}

/// The drivers' gains in flat arrays (structure of arrays), for the CPU expression sum and the GPU
/// activation table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DriverTable {
    regions: Vec<u32>,
    compression_gains: Vec<f32>,
    stretch_gains: Vec<f32>,
    offsets: Vec<u32>,
    expressions: Vec<u32>,
    gains: Vec<f32>,
}

impl DriverTable {
    /// The table of `drivers` (the same list `WrinkleDrivers::new` checks), in the same order.
    pub fn new(drivers: &[WrinkleDriver]) -> Result<Self, SkinError> {
        if drivers.len() > MAX_WRINKLE_DRIVERS {
            return Err(SkinError::limit("wrinkle drivers", MAX_WRINKLE_DRIVERS as u64, drivers.len() as u64));
        }
        let bounded = |g: f32| g.is_finite() && g.abs() <= MAX_GAIN;
        let mut out = Self { offsets: vec![0], ..Self::default() };
        for d in drivers {
            if d.expressions.len() > MAX_DRIVER_EXPRESSIONS {
                return Err(SkinError::limit(
                    "wrinkle driver expressions",
                    MAX_DRIVER_EXPRESSIONS as u64,
                    d.expressions.len() as u64,
                ));
            }
            if !(bounded(d.compression_gain) && bounded(d.stretch_gain) && d.expressions.iter().all(|e| bounded(e.1))) {
                return Err(SkinError::Invalid { what: "wrinkle driver gain not finite or above MAX_GAIN" });
            }
            out.regions.push(u32::from(d.region));
            out.compression_gains.push(d.compression_gain);
            out.stretch_gains.push(d.stretch_gain);
            for &(e, g) in &d.expressions {
                out.expressions.push(e);
                out.gains.push(g);
            }
            out.offsets.push(out.expressions.len() as u32);
        }
        Ok(out)
    }

    /// How many drivers there are.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether there are no drivers.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The expression part of every driver's fold (unclamped; missing or non-finite weights count
    /// as 0) into `out[..len]`. Allocation-free.
    pub fn expression_terms(&self, weights: &[f32], out: &mut [f32]) -> Result<(), SkinError> {
        if out.len() < self.len() {
            return Err(SkinError::Invalid { what: "one expression term per driver" });
        }
        for (d, o) in out.iter_mut().take(self.len()).enumerate() {
            let range = self.offsets[d] as usize..self.offsets[d + 1] as usize;
            let mut sum = 0.0;
            for (&e, &g) in self.expressions[range.clone()].iter().zip(&self.gains[range]) {
                sum += g * weights.get(e as usize).copied().filter(|w| w.is_finite()).unwrap_or(0.0);
            }
            *o = sum;
        }
        Ok(())
    }

    /// The activations from the expression terms and the per-region strain (missing entries count
    /// as 0): the CPU twin of `activate` in `skin/wrinkle_drivers.wgsl`.
    pub fn activate(
        &self,
        terms: &[f32],
        region_stretch: &[f32],
        region_compression: &[f32],
        fold: &mut [f32],
        stretch: &mut [f32],
    ) -> Result<(), SkinError> {
        if fold.len() < self.len() || stretch.len() < self.len() {
            return Err(SkinError::Invalid { what: "one fold and one stretch output per driver" });
        }
        let read = |a: &[f32], i: usize| a.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
        for d in 0..self.len() {
            let r = self.regions[d] as usize;
            let f = read(terms, d) + self.compression_gains[d] * read(region_compression, r);
            fold[d] = if f.is_finite() { f.clamp(0.0, 1.0) } else { 0.0 };
            let s = self.stretch_gains[d] * read(region_stretch, r);
            stretch[d] = if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 };
        }
        Ok(())
    }

    /// The GPU table: per driver `(region, compression gain bits, stretch gain bits, 0)`.
    pub fn gpu_words(&self, out: &mut [[u32; 4]]) -> Result<(), SkinError> {
        if out.len() < self.len() {
            return Err(SkinError::Invalid { what: "one GPU record per driver" });
        }
        for (d, o) in out.iter_mut().take(self.len()).enumerate() {
            *o = [self.regions[d], self.compression_gains[d].to_bits(), self.stretch_gains[d].to_bits(), 0];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::WrinkleDrivers;

    fn map(driver: u16, rect: Rect, h: impl Fn(u32, u32) -> f32, flatten: f32) -> WrinkleMap {
        let mut height = Vec::new();
        for y in 0..rect.h {
            for x in 0..rect.w {
                height.push(h(x, y));
            }
        }
        let cavity = height.iter().map(|h| 0.5 * h).collect();
        WrinkleMap { driver, tile: 2, rect, height, cavity, flatten }
    }

    #[test]
    fn skn30_activations_match_the_wrinkle_drivers() {
        let drivers = vec![
            WrinkleDriver {
                name: "a".into(),
                region: 1,
                expressions: vec![(0, 0.5), (2, -0.25)],
                compression_gain: 8.0,
                stretch_gain: 6.0,
            },
            WrinkleDriver {
                name: "b".into(),
                region: 0,
                expressions: vec![],
                compression_gain: -3.0,
                stretch_gain: 40.0,
            },
            WrinkleDriver {
                name: "c".into(),
                region: 2,
                expressions: vec![(1, 2.0)],
                compression_gain: 0.0,
                stretch_gain: 0.0,
            },
        ];
        let reference = WrinkleDrivers::new(&drivers, 3, 3).unwrap();
        let table = DriverTable::new(&drivers).unwrap();
        let weights = [0.8, 0.3, f32::NAN];
        let stretch_r = [0.01, 0.2, 0.05];
        let compression_r = [0.02, 0.05, 0.0];
        let (mut f0, mut s0) = ([0.0; 3], [0.0; 3]);
        reference.evaluate(&weights, &stretch_r, &compression_r, &mut f0, &mut s0).unwrap();
        let mut terms = [0.0; 3];
        table.expression_terms(&weights, &mut terms).unwrap();
        let (mut f1, mut s1) = ([0.0; 3], [0.0; 3]);
        table.activate(&terms, &stretch_r, &compression_r, &mut f1, &mut s1).unwrap();
        assert_eq!(f0, f1);
        assert_eq!(s0, s1);
        assert!(f1[0] > 0.0 && f1[0] < 1.0 && s1[1] > 0.0);
        let mut words = [[0; 4]; 3];
        table.gpu_words(&mut words).unwrap();
        assert_eq!(words[1], [0, (-3.0_f32).to_bits(), 40.0_f32.to_bits(), 0]);
        assert!(table.expression_terms(&weights, &mut [0.0; 2]).is_err());
        let bad = [WrinkleDriver { compression_gain: f32::INFINITY, ..drivers[0].clone() }];
        assert!(DriverTable::new(&bad).is_err());
    }

    #[test]
    fn skn30_bake_slots_slopes_and_blend() {
        // Driver 0: a horizontal crease (height varies with y); driver 1: a ramp along x over an
        // overlapping rectangle; driver 2: weaker, overlapping both.
        let crease = map(0, Rect::new(4, 4, 16, 16), |_, y| -0.1 * (y as f32 - 8.0).abs().min(4.0), 0.5);
        let ramp = map(1, Rect::new(8, 8, 16, 16), |x, _| 0.01 * x as f32, 1.0);
        let weak = map(2, Rect::new(10, 10, 4, 4), |_, _| 1e-4, 0.2);
        let maps = WrinkleMaps::bake(32, &[2], &[0.5], &[crease.clone(), ramp.clone(), weak]).unwrap();
        assert_eq!(maps.tiles(), [2]);
        // Outside every map: empty.
        assert_eq!(maps.texel(2, 0, 0), Some(&WrinkleTexel::default()));
        assert_eq!(maps.texel(1, 0, 0), None);
        // Inside the crease only.
        let t = maps.texel(2, 6, 10).unwrap();
        assert_eq!(t.id, [1, 0]);
        assert_eq!(t.height[0], crease.height_at(2, 6));
        // Central difference along v over 2 texels of 0.5 mm.
        let sv = (crease.height_at(2, 7) - crease.height_at(2, 5)) / 1.0;
        assert!((sv - 0.2).abs() < 1e-6);
        assert_eq!(t.slope[0], [0.0, sv]);
        assert_eq!(t.cavity[0], 0.5 * t.height[0]);
        // Overlap of all three: the weak driver lost to the two strong ones.
        let t = maps.texel(2, 12, 11).unwrap();
        assert_eq!(t.id, [1, 2]);
        assert_eq!(t.flatten, [0.5, 1.0]);
        // Blend: fold scales heights, stretch flattens detail.
        let s = blend_texel(t, &[0.5, 1.0, 1.0], &[0.2, 0.4, 1.0]);
        assert!((s.height - (0.5 * t.height[0] + t.height[1])).abs() < 1e-7);
        assert!((s.slope[0] - (0.5 * t.slope[0][0] + t.slope[1][0])).abs() < 1e-7);
        assert!((s.detail_scale - (1.0 - 0.2 * 0.5 - 0.4)).abs() < 1e-6);
        // No activations: nothing changes.
        assert_eq!(blend_texel(t, &[], &[]), WrinkleSample { detail_scale: 1.0, ..WrinkleSample::default() });
        // Two maps of one driver add into one slot.
        let twice = WrinkleMaps::bake(32, &[2], &[0.5], &[ramp.clone(), ramp.clone()]).unwrap();
        let t = twice.texel(2, 12, 12).unwrap();
        assert_eq!(t.id, [2, 0]);
        assert!((t.height[0] - 2.0 * ramp.height_at(4, 4)).abs() < 1e-7);
        // GPU layers.
        let mut layer = vec![[0.0; 4]; 32 * 32];
        maps.gpu_layer(0, 2, &mut layer).unwrap();
        assert_eq!(layer[11 * 32 + 12], [1.0, 2.0, 0.5, 1.0]);
        maps.gpu_layer(0, 1, &mut layer).unwrap();
        let t = maps.texel(2, 12, 11).unwrap();
        assert_eq!(layer[11 * 32 + 12], [t.height[1], t.slope[1][0], t.slope[1][1], t.cavity[1]]);
        assert!(maps.gpu_layer(1, 0, &mut layer).is_err());
        assert!(maps.gpu_layer(0, 3, &mut layer).is_err());
    }

    #[test]
    fn wrinkle_bake_input_checks() {
        let m = map(0, Rect::new(0, 0, 4, 4), |_, _| 0.1, 0.5);
        assert!(WrinkleMaps::bake(0, &[2], &[0.5], &[]).is_err());
        assert!(WrinkleMaps::bake(MAX_WRINKLE_SIZE + 1, &[2], &[0.5], &[]).is_err());
        assert!(WrinkleMaps::bake(8, &[2, 2], &[0.5, 0.5], &[]).is_err());
        assert!(WrinkleMaps::bake(8, &[2], &[0.0], &[]).is_err());
        assert!(WrinkleMaps::bake(8, &[3], &[0.5], std::slice::from_ref(&m)).is_err());
        assert!(WrinkleMaps::bake(2, &[2], &[0.5], std::slice::from_ref(&m)).is_err());
        let bad = WrinkleMap { flatten: 2.0, ..m.clone() };
        assert!(WrinkleMaps::bake(8, &[2], &[0.5], &[bad]).is_err());
        let bad = WrinkleMap { height: vec![0.0; 3], ..m.clone() };
        assert!(WrinkleMaps::bake(8, &[2], &[0.5], &[bad]).is_err());
        let bad = WrinkleMap { driver: MAX_WRINKLE_DRIVERS as u16, ..m };
        assert!(WrinkleMaps::bake(8, &[2], &[0.5], &[bad]).is_err());
    }

    #[test]
    fn skn30_maps_from_the_wrinkle_generator() {
        use super::super::features::{GeneratorKind, GeneratorParams, Target, generate};
        use super::super::layers::{LayerId, LayerKind};
        let params = GeneratorParams::new(GeneratorKind::Wrinkles);
        let target = Target { tile: 2, rect: Rect::new(16, 16, 96, 64), texel_mm: 0.25, mask: None };
        let layer = generate(&params, &target, LayerId(1), 7, None).unwrap();
        let LayerKind::Content { tiles, .. } = &layer.kind else { panic!("content layer") };
        let m = WrinkleMap::from_content(3, &tiles[0], 0.6).unwrap();
        assert!(m.height.iter().any(|h| *h != 0.0));
        let maps = WrinkleMaps::bake(128, &[2], &[0.25], &[m]).unwrap();
        let (x, y) = (16..112)
            .flat_map(|x| (16..80).map(move |y| (x, y)))
            .max_by(|a, b| {
                let h = |p: &(u32, u32)| maps.texel(2, p.0, p.1).map_or(0.0, |t| t.height[0].abs());
                h(a).total_cmp(&h(b))
            })
            .unwrap();
        let t = maps.texel(2, x, y).unwrap();
        assert_eq!(t.id, [4, 0]);
        assert!(t.height[0] != 0.0);
        // Half a fold gives half the depth; other drivers do nothing here.
        assert_eq!(blend_texel(t, &[1.0, 1.0, 1.0, 0.5], &[]).height, 0.5 * t.height[0]);
    }

    #[test]
    fn perf_r1_per_frame_driver_work_allocates_nothing() {
        // [PERF-R1] The per-frame CPU half of [SKN-30] (expression terms, and the CPU twin of the
        // activation) allocates nothing; the counting allocator is the test binary one (eval/tests.rs).
        assert!(humanart_core::alloc::is_installed());
        let drivers: Vec<WrinkleDriver> = (0..64)
            .map(|i| WrinkleDriver {
                name: format!("d{i}"),
                region: (i % 7) as u16,
                expressions: vec![(i % 5, 0.5), ((i + 1) % 5, -0.1)],
                compression_gain: 4.0,
                stretch_gain: 3.0,
            })
            .collect();
        let table = DriverTable::new(&drivers).unwrap();
        let (mut terms, mut fold, mut stretch) = ([0.0; 64], [0.0; 64], [0.0; 64]);
        let weights = [0.1, 0.5, 0.9, 0.0, 1.0];
        let strain = [0.05; 7];
        let stats = humanart_core::alloc::count_in(|| {
            for _ in 0..100 {
                table.expression_terms(&weights, &mut terms).unwrap();
                table.activate(&terms, &strain, &strain, &mut fold, &mut stretch).unwrap();
            }
        });
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert!(fold.iter().all(|f| *f > 0.0));
    }
}
