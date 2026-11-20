//! Groom generation parameters and maps `[HAIR-02]` `[DOC-08]`.
//!
//! [`GroomParams`] holds every modifier setting as a flat `f32` with a stable key, a range and a
//! default ([`PARAMS`]); a document groom stores them as its `params` object (key to number), so
//! [`GroomParams::set`] / [`GroomParams::entries`] are the bridge. Keys are stable ids: never
//! renamed or reused. Lengths and radii are metres, frequencies per metre, angles radians.
//!
//! [`GroomMaps`] are the painted UV maps of a groom (`density`, `length`, `direction`, `clump`,
//! `curl`, `frizz`, `color`, the names of the document's `GROOM_MAPS`). Each is a small `f32` grid
//! sampled bilinearly at the strand root's UV (origin top-left, V down).

use humanart_math::{Vec2, Vec3};

use super::error::{HairError, HairResult};

/// One scalar parameter: key, range, default, meaning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    /// The stable key.
    pub key: &'static str,
    /// Smallest accepted value.
    pub min: f32,
    /// Largest accepted value.
    pub max: f32,
    /// The default.
    pub default: f32,
    /// What it does.
    pub doc: &'static str,
}

macro_rules! groom_params {
    ($($field:ident: $key:literal, $min:expr, $max:expr, $default:expr, $doc:literal;)*) => {
        /// Every generation setting of one groom. See the module documentation.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct GroomParams {
            $(#[doc = $doc] pub $field: f32,)*
        }

        /// Every parameter of [`GroomParams`] with its key, range and default.
        pub const PARAMS: &[ParamSpec] = &[
            $(ParamSpec { key: $key, min: $min, max: $max, default: $default, doc: $doc },)*
        ];

        impl Default for GroomParams {
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        impl GroomParams {
            /// The value under `key`.
            pub fn get(&self, key: &str) -> Option<f32> {
                match key {
                    $($key => Some(self.$field),)*
                    _ => None,
                }
            }

            fn slot(&mut self, key: &str) -> Option<&mut f32> {
                match key {
                    $($key => Some(&mut self.$field),)*
                    _ => None,
                }
            }
        }
    };
}

groom_params! {
    strand_count: "strands.count", 0.0, 4_194_304.0, 100_000.0, "Strands generated (rounded down).";
    points: "strands.points", 2.0, 32.0, 16.0, "Control points per generated strand.";
    interpolation: "strands.guides", 1.0, 4.0, 3.0, "Guides blended per strand.";
    length_scale: "length.scale", 0.0, 4.0, 1.0, "Multiplier on the interpolated guide length.";
    length_variation: "length.variation", 0.0, 1.0, 0.1, "Random per-strand length change, +- fraction.";
    direction_strength: "direction.strength", 0.0, 1.0, 1.0, "How far the direction map turns strands.";
    noise: "noise", 0.0, 3.0, 0.05, "Random per-strand turn about the root normal, +- radians.";
    curl_radius: "curl.radius", 0.0, 0.05, 0.0, "Curl helix radius.";
    curl_frequency: "curl.frequency", 0.0, 400.0, 20.0, "Curl turns per metre.";
    curl_variation: "curl.variation", 0.0, 1.0, 0.2, "Random per-strand curl radius and frequency change.";
    wave_amplitude: "wave.amplitude", 0.0, 0.05, 0.0, "Planar wave amplitude.";
    wave_frequency: "wave.frequency", 0.0, 200.0, 8.0, "Wave periods per metre.";
    frizz: "frizz", 0.0, 0.02, 0.0005, "Random displacement of points, growing to the tip.";
    flyaway_fraction: "flyaway.fraction", 0.0, 1.0, 0.01, "Share of strands that stray from the groom.";
    flyaway_strength: "flyaway.strength", 0.0, 0.05, 0.01, "How far flyaway tips stray.";
    clump_count: "clump.count", 0.0, 65_536.0, 400.0, "Clump centres.";
    clump_strength: "clump.strength", 0.0, 1.0, 0.4, "How strongly strands are pulled to their clump centre.";
    clump_spread: "clump.spread", 0.0, 1.0, 0.6, "Share of the distance to the centre kept near the root.";
    clump_tip_spread: "clump.tip_spread", 0.0, 1.0, 0.2, "Share of the distance to the centre kept at the tip.";
    hairline_softness: "hairline.softness", 0.0, 0.05, 0.008, "Width of the density fade at the hairline.";
    hairline_irregularity: "hairline.irregularity", 0.0, 1.0, 0.5, "Noise on the hairline, share of the softness.";
    thickness: "thickness", 0.000_001, 0.000_5, 0.000_07, "Fibre diameter at the root.";
    tip_ratio: "thickness.tip_ratio", 0.0, 1.0, 0.25, "Tip diameter over root diameter.";
    thickness_variation: "thickness.variation", 0.0, 1.0, 0.2, "Random per-strand diameter change, +- fraction.";
    color_variation: "color.variation", 0.0, 1.0, 0.1, "Strength of the per-strand colour variation (shader).";
    grey_fraction: "grey.fraction", 0.0, 1.0, 0.0, "Share of unpigmented strands (age greying).";
    stiffness: "stiffness", 0.0, 1.0, 0.5, "Stiffness hint for game-side dynamics.";
    collision_offset: "collision.offset", 0.0, 0.02, 0.002, "Distance strand points are kept above the skin.";
    parting_x: "parting.x", -10.0, 10.0, 0.0, "A point of the parting plane, x.";
    parting_y: "parting.y", -10.0, 10.0, 0.0, "A point of the parting plane, y.";
    parting_z: "parting.z", -10.0, 10.0, 0.0, "A point of the parting plane, z.";
    parting_nx: "parting.nx", -1.0, 1.0, 1.0, "The parting plane normal, x.";
    parting_ny: "parting.ny", -1.0, 1.0, 0.0, "The parting plane normal, y.";
    parting_nz: "parting.nz", -1.0, 1.0, 0.0, "The parting plane normal, z.";
    parting_width: "parting.width", 0.0, 0.1, 0.02, "Width of the band either side of the parting that is combed apart.";
    parting_strength: "parting.strength", 0.0, 1.0, 0.0, "Parting strength; 0 is no parting.";
    cowlick0_x: "cowlick0.x", -10.0, 10.0, 0.0, "Cowlick 0 centre, x.";
    cowlick0_y: "cowlick0.y", -10.0, 10.0, 0.0, "Cowlick 0 centre, y.";
    cowlick0_z: "cowlick0.z", -10.0, 10.0, 0.0, "Cowlick 0 centre, z.";
    cowlick0_radius: "cowlick0.radius", 0.0, 0.1, 0.0, "Cowlick 0 radius; 0 is off.";
    cowlick0_strength: "cowlick0.strength", -1.0, 1.0, 0.0, "Cowlick 0 swirl, sign is the turning sense.";
    cowlick1_x: "cowlick1.x", -10.0, 10.0, 0.0, "Cowlick 1 centre, x.";
    cowlick1_y: "cowlick1.y", -10.0, 10.0, 0.0, "Cowlick 1 centre, y.";
    cowlick1_z: "cowlick1.z", -10.0, 10.0, 0.0, "Cowlick 1 centre, z.";
    cowlick1_radius: "cowlick1.radius", 0.0, 0.1, 0.0, "Cowlick 1 radius; 0 is off.";
    cowlick1_strength: "cowlick1.strength", -1.0, 1.0, 0.0, "Cowlick 1 swirl, sign is the turning sense.";
    cowlick2_x: "cowlick2.x", -10.0, 10.0, 0.0, "Cowlick 2 centre, x.";
    cowlick2_y: "cowlick2.y", -10.0, 10.0, 0.0, "Cowlick 2 centre, y.";
    cowlick2_z: "cowlick2.z", -10.0, 10.0, 0.0, "Cowlick 2 centre, z.";
    cowlick2_radius: "cowlick2.radius", 0.0, 0.1, 0.0, "Cowlick 2 radius; 0 is off.";
    cowlick2_strength: "cowlick2.strength", -1.0, 1.0, 0.0, "Cowlick 2 swirl, sign is the turning sense.";
    cowlick3_x: "cowlick3.x", -10.0, 10.0, 0.0, "Cowlick 3 centre, x.";
    cowlick3_y: "cowlick3.y", -10.0, 10.0, 0.0, "Cowlick 3 centre, y.";
    cowlick3_z: "cowlick3.z", -10.0, 10.0, 0.0, "Cowlick 3 centre, z.";
    cowlick3_radius: "cowlick3.radius", 0.0, 0.1, 0.0, "Cowlick 3 radius; 0 is off.";
    cowlick3_strength: "cowlick3.strength", -1.0, 1.0, 0.0, "Cowlick 3 swirl, sign is the turning sense.";
    eumelanin: "melanin.eumelanin", 0.0, 16.0, 1.3, "Eumelanin concentration (pbrt scale; about 0.3 blonde, 1.3 brown, 8 black).";
    pheomelanin: "melanin.pheomelanin", 0.0, 16.0, 0.0, "Pheomelanin concentration (pbrt scale); orange to red.";
    dye_r: "dye.r", 0.001, 1.0, 0.5, "Dye layer target colour, linear red.";
    dye_g: "dye.g", 0.001, 1.0, 0.5, "Dye layer target colour, linear green.";
    dye_b: "dye.b", 0.001, 1.0, 0.5, "Dye layer target colour, linear blue.";
    dye_amount: "dye.amount", 0.0, 1.0, 0.0, "Dye layer opacity; 0 is natural colour.";
    azimuthal_roughness: "shading.azimuthal_roughness", 0.0, 1.0, 0.3, "Azimuthal roughness beta_n of the fibre shader, used to invert the dye colour.";
}

/// The number of cowlicks a groom may have.
pub const MAX_COWLICKS: usize = 4;

/// A parting: strands within `width` of the plane through `point` with unit `normal` are turned
/// away from it with `strength`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parting {
    /// A point of the plane.
    pub point: Vec3,
    /// The unit plane normal.
    pub normal: Vec3,
    /// Half-width of the affected band.
    pub width: f32,
    /// 0 (off) to 1.
    pub strength: f32,
}

/// A cowlick: strands within `radius` of `centre` swirl around it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cowlick {
    /// The centre (on the scalp).
    pub centre: Vec3,
    /// The radius; 0 is off.
    pub radius: f32,
    /// -1..1, the sign is the turning sense about the surface normal.
    pub strength: f32,
}

impl GroomParams {
    /// Sets `key` to `value`; unknown keys and values outside the range (or not finite) are errors.
    pub fn set(&mut self, key: &str, value: f64) -> HairResult<()> {
        let spec = PARAMS.iter().find(|s| s.key == key).ok_or_else(|| HairError::Param {
            name: key.chars().take(64).collect(),
            what: "unknown groom parameter",
        })?;
        let v = value as f32;
        if !v.is_finite() || v < spec.min || v > spec.max {
            return Err(HairError::Param { name: key.into(), what: "value outside the parameter's range" });
        }
        if let Some(slot) = self.slot(key) {
            *slot = v;
        }
        Ok(())
    }

    /// The parameters from document entries (key, value); keys not listed keep their default.
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a str, f64)>) -> HairResult<Self> {
        let mut p = Self::default();
        for (k, v) in entries {
            p.set(k, v)?;
        }
        Ok(p)
    }

    /// The parameters that differ from their default, as (key, value): what a document stores.
    pub fn entries(&self) -> Vec<(&'static str, f64)> {
        PARAMS
            .iter()
            .filter_map(|s| self.get(s.key).filter(|&v| v != s.default).map(|v| (s.key, f64::from(v))))
            .collect()
    }

    /// Checks every value against its range.
    pub fn check(&self) -> HairResult<()> {
        for s in PARAMS {
            let v = self.get(s.key).unwrap_or(s.default);
            if !v.is_finite() || v < s.min || v > s.max {
                return Err(HairError::Param { name: s.key.into(), what: "value outside the parameter's range" });
            }
        }
        Ok(())
    }

    /// Strands to generate.
    pub fn strand_count(&self) -> u32 {
        self.strand_count.max(0.0) as u32
    }

    /// Control points per generated strand (2..=32).
    pub fn point_count(&self) -> u32 {
        (self.points as u32).clamp(2, 32)
    }

    /// Guides blended per strand (1..=4).
    pub fn guides_per_strand(&self) -> usize {
        (self.interpolation as usize).clamp(1, 4)
    }

    /// Clump centres.
    pub fn clump_centres(&self) -> u32 {
        self.clump_count.max(0.0) as u32
    }

    /// The parting.
    pub fn parting(&self) -> Parting {
        Parting {
            point: Vec3::new(self.parting_x, self.parting_y, self.parting_z),
            normal: Vec3::new(self.parting_nx, self.parting_ny, self.parting_nz).normalize_or(Vec3::X),
            width: self.parting_width,
            strength: self.parting_strength,
        }
    }

    /// Sets the parting.
    pub fn set_parting(&mut self, p: Parting) {
        let n = p.normal.normalize_or(Vec3::X);
        (self.parting_x, self.parting_y, self.parting_z) = (p.point.x, p.point.y, p.point.z);
        (self.parting_nx, self.parting_ny, self.parting_nz) = (n.x, n.y, n.z);
        self.parting_width = p.width.clamp(0.0, 0.1);
        self.parting_strength = p.strength.clamp(0.0, 1.0);
    }

    /// Cowlick `i` (`None` above [`MAX_COWLICKS`]).
    pub fn cowlick(&self, i: usize) -> Option<Cowlick> {
        let c = match i {
            0 => [self.cowlick0_x, self.cowlick0_y, self.cowlick0_z, self.cowlick0_radius, self.cowlick0_strength],
            1 => [self.cowlick1_x, self.cowlick1_y, self.cowlick1_z, self.cowlick1_radius, self.cowlick1_strength],
            2 => [self.cowlick2_x, self.cowlick2_y, self.cowlick2_z, self.cowlick2_radius, self.cowlick2_strength],
            3 => [self.cowlick3_x, self.cowlick3_y, self.cowlick3_z, self.cowlick3_radius, self.cowlick3_strength],
            _ => return None,
        };
        Some(Cowlick { centre: Vec3::new(c[0], c[1], c[2]), radius: c[3], strength: c[4] })
    }

    /// Sets cowlick `i`; `false` above [`MAX_COWLICKS`].
    pub fn set_cowlick(&mut self, i: usize, c: Cowlick) -> bool {
        let slots = match i {
            0 => [
                &mut self.cowlick0_x,
                &mut self.cowlick0_y,
                &mut self.cowlick0_z,
                &mut self.cowlick0_radius,
                &mut self.cowlick0_strength,
            ],
            1 => [
                &mut self.cowlick1_x,
                &mut self.cowlick1_y,
                &mut self.cowlick1_z,
                &mut self.cowlick1_radius,
                &mut self.cowlick1_strength,
            ],
            2 => [
                &mut self.cowlick2_x,
                &mut self.cowlick2_y,
                &mut self.cowlick2_z,
                &mut self.cowlick2_radius,
                &mut self.cowlick2_strength,
            ],
            3 => [
                &mut self.cowlick3_x,
                &mut self.cowlick3_y,
                &mut self.cowlick3_z,
                &mut self.cowlick3_radius,
                &mut self.cowlick3_strength,
            ],
            _ => return false,
        };
        let values = [c.centre.x, c.centre.y, c.centre.z, c.radius.clamp(0.0, 0.1), c.strength.clamp(-1.0, 1.0)];
        for (slot, v) in slots.into_iter().zip(values) {
            *slot = v;
        }
        true
    }
}

/// The largest groom map side.
pub const MAX_MAP_SIDE: u32 = 4096;

/// A painted groom map: `channels` `f32` values per texel, row-major from the top-left.
#[derive(Clone, Debug, PartialEq)]
pub struct GroomMap {
    width: u32,
    height: u32,
    channels: u32,
    data: Vec<f32>,
}

impl GroomMap {
    /// A `width` x `height` map with `channels` (1..=4) channels, every texel `fill`.
    pub fn new(width: u32, height: u32, channels: u32, fill: &[f32]) -> HairResult<Self> {
        if width == 0 || height == 0 || width > MAX_MAP_SIDE || height > MAX_MAP_SIDE {
            return Err(HairError::limit("groom map side", u64::from(MAX_MAP_SIDE), u64::from(width.max(height))));
        }
        if !(1..=4).contains(&channels) || fill.len() != channels as usize || fill.iter().any(|v| !v.is_finite()) {
            return Err(HairError::invalid("groom map", 0, "1 to 4 channels with a finite fill value each"));
        }
        let mut data = Vec::with_capacity(width as usize * height as usize * channels as usize);
        for _ in 0..width as usize * height as usize {
            data.extend_from_slice(fill);
        }
        Ok(Self { width, height, channels, data })
    }

    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Channels per texel.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// The texels.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The texel at `(x, y)`, writable (`None` outside).
    pub fn texel_mut(&mut self, x: u32, y: u32) -> Option<&mut [f32]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let c = self.channels as usize;
        let at = (y as usize * self.width as usize + x as usize) * c;
        self.data.get_mut(at..at + c)
    }

    /// The texel at `(x, y)` clamped to the map.
    pub fn texel(&self, x: i64, y: i64) -> &[f32] {
        let x = x.clamp(0, i64::from(self.width) - 1) as usize;
        let y = y.clamp(0, i64::from(self.height) - 1) as usize;
        let c = self.channels as usize;
        let at = (y * self.width as usize + x) * c;
        self.data.get(at..at + c).unwrap_or(&[])
    }

    /// Channel `channel` bilinearly sampled at `uv` (texel centres at half-integers, clamped).
    pub fn sample(&self, uv: Vec2, channel: u32) -> f32 {
        let x = uv.x * self.width as f32 - 0.5;
        let y = uv.y * self.height as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let c = channel as usize;
        let t = |dx: i64, dy: i64| self.texel(x0 as i64 + dx, y0 as i64 + dy).get(c).copied().unwrap_or(0.0);
        let top = t(0, 0) + (t(1, 0) - t(0, 0)) * fx;
        let bottom = t(0, 1) + (t(1, 1) - t(0, 1)) * fx;
        top + (bottom - top) * fy
    }
}

/// The painted maps of a groom; a missing map is neutral.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroomMaps {
    /// Root density, 0..1 (1 channel; neutral 1).
    pub density: Option<GroomMap>,
    /// Length multiplier (1 channel; neutral 1).
    pub length: Option<GroomMap>,
    /// Flow in the root frame's `(t, b)` plane, its length the strength (2 channels; neutral 0).
    pub direction: Option<GroomMap>,
    /// Clump strength multiplier (1 channel; neutral 1).
    pub clump: Option<GroomMap>,
    /// Curl radius multiplier (1 channel; neutral 1).
    pub curl: Option<GroomMap>,
    /// Frizz multiplier (1 channel; neutral 1).
    pub frizz: Option<GroomMap>,
    /// Colour offset -1..1 handed to the shader (1 channel; neutral 0).
    pub color: Option<GroomMap>,
}

impl GroomMaps {
    fn at(map: Option<&GroomMap>, uv: Vec2, channel: u32, neutral: f32) -> f32 {
        map.map_or(neutral, |m| m.sample(uv, channel))
    }

    /// The density at `uv`, clamped to 0..1.
    pub fn density_at(&self, uv: Vec2) -> f32 {
        Self::at(self.density.as_ref(), uv, 0, 1.0).clamp(0.0, 1.0)
    }

    /// The per-strand factors at `uv`: length, clump, curl, frizz multipliers (>= 0), direction
    /// flow `(x, y)` and colour offset.
    pub fn factors_at(&self, uv: Vec2) -> ([f32; 4], [f32; 2], f32) {
        let m = [
            Self::at(self.length.as_ref(), uv, 0, 1.0).max(0.0),
            Self::at(self.clump.as_ref(), uv, 0, 1.0).max(0.0),
            Self::at(self.curl.as_ref(), uv, 0, 1.0).max(0.0),
            Self::at(self.frizz.as_ref(), uv, 0, 1.0).max(0.0),
        ];
        let d = [Self::at(self.direction.as_ref(), uv, 0, 0.0), Self::at(self.direction.as_ref(), uv, 1, 0.0)];
        (m, d, Self::at(self.color.as_ref(), uv, 0, 0.0).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_round_trip_through_entries() {
        let mut p = GroomParams::default();
        assert!(p.check().is_ok());
        assert!(p.entries().is_empty());
        p.set("curl.radius", 0.004).unwrap();
        p.set("cowlick2.radius", 0.03).unwrap();
        assert!(p.set("curl.radius", 1.0).is_err());
        assert!(p.set("curl.radius", f64::NAN).is_err());
        assert!(p.set("no.such", 1.0).is_err());
        let e = p.entries();
        assert_eq!(e.len(), 2);
        let q = GroomParams::from_entries(e.iter().map(|&(k, v)| (k, v))).unwrap();
        assert_eq!(p, q);
        assert!((q.cowlick(2).unwrap().radius - 0.03).abs() < 1e-7);
        assert!(q.cowlick(4).is_none());
        // Keys are unique and defaults are inside their ranges.
        let mut keys: Vec<_> = PARAMS.iter().map(|s| s.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), PARAMS.len());
        assert!(PARAMS.iter().all(|s| s.min <= s.default && s.default <= s.max));
    }

    #[test]
    fn parting_and_cowlicks_set_and_read_back() {
        let mut p = GroomParams::default();
        p.set_parting(Parting { point: Vec3::Y, normal: Vec3::new(2.0, 0.0, 0.0), width: 0.03, strength: 0.7 });
        let q = p.parting();
        assert_eq!(q.normal, Vec3::X);
        assert_eq!(q.point, Vec3::Y);
        assert!(p.set_cowlick(3, Cowlick { centre: Vec3::Z, radius: 0.02, strength: -0.5 }));
        assert!(!p.set_cowlick(4, Cowlick { centre: Vec3::Z, radius: 0.02, strength: -0.5 }));
        assert_eq!(p.cowlick(3).unwrap().strength, -0.5);
        assert!(p.check().is_ok());
    }

    #[test]
    fn maps_sample_bilinearly_and_check_input() {
        let mut m = GroomMap::new(2, 1, 1, &[0.0]).unwrap();
        m.texel_mut(1, 0).unwrap()[0] = 1.0;
        assert!((m.sample(Vec2::new(0.5, 0.5), 0) - 0.5).abs() < 1e-6);
        assert_eq!(m.sample(Vec2::new(0.0, 0.0), 0), 0.0);
        assert_eq!(m.sample(Vec2::new(1.0, 1.0), 0), 1.0);
        assert!(m.texel_mut(2, 0).is_none());
        assert!(GroomMap::new(0, 1, 1, &[0.0]).is_err());
        assert!(GroomMap::new(1, 1, 2, &[0.0]).is_err());
        let maps = GroomMaps { density: Some(m), ..GroomMaps::default() };
        assert_eq!(maps.density_at(Vec2::new(1.0, 0.0)), 1.0);
        let (f, d, c) = maps.factors_at(Vec2::ZERO);
        assert_eq!((f, d, c), ([1.0; 4], [0.0; 2], 0.0));
    }
}
