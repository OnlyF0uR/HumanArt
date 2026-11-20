//! Scan albedo into the template's UV layout, with de-lighting `[WP-25]` (todo section 3).
//!
//! **Transfer.** Each fitted template triangle of one UV tile is rasterised in UV space; at each
//! texel centre the fitted surface point (and its interpolated normal) is looked up on the scan:
//! the closest scan point within [`TransferSettings::max_distance`] whose triangle faces the same
//! way (so the inside of a lip does not pick the other lip), its scan UV interpolated
//! barycentrically, the scan albedo sampled there. The distance is kept per texel: it is the fit
//! error painted into the template's layout.
//!
//! **De-lighting.** A photogrammetry texture carries the light of the capture rig. Two parts are
//! removed:
//!
//! - **baked occlusion**: the rig is close to uniform ambient light, so a texel's brightness is
//!   about albedo × visibility. Our own ambient occlusion of the scan ([`scan_occlusion`], the
//!   cosine-weighted visible fraction of the hemisphere at each scan vertex) is divided out:
//!   `albedo = observed / max(1 − s (1 − ao), floor)` with strength `s` and a floor that keeps deep
//!   cavities (nostrils) from blowing up;
//! - **specular specks**: texels much brighter than their surroundings and nearly colourless (the
//!   highlights the cross-polarisation missed, reconstruction errors) are found against a local
//!   mean ([`find_specks`]) and inpainted from their neighbours ([`inpaint`]).
//!
//! Finally empty texels next to islands are filled outward ([`TransferredAlbedo::pad`]) so
//! filtering does not pull black over seams.

use humanart_math::{Vec2, Vec3};

use super::albedo::Albedo;
use super::nearest::closest_on_triangle;
use super::register::TargetSurface;
use super::scan::ScanMesh;
use super::{CaptureError, Result};
use crate::bake::bvh::Bvh;
use crate::bake::sampling::{cosine_hemisphere, point, rotation, texel_hash, to_world};

/// Largest output side.
pub const MAX_TRANSFER_SIDE: u32 = 8192;
/// Most occlusion rays per scan vertex.
pub const MAX_OCCLUSION_RAYS: u32 = 1024;

/// Texel states of a [`TransferredAlbedo`].
pub mod state {
    /// No surface covers the texel.
    pub const EMPTY: u8 = 0;
    /// Taken from the scan.
    pub const SCAN: u8 = 1;
    /// A speck, inpainted from its neighbours.
    pub const INPAINTED: u8 = 2;
    /// Covered by the template but no scan point within reach: inpainted.
    pub const NO_SCAN: u8 = 3;
    /// Padding outside the islands.
    pub const PADDING: u8 = 4;
}

/// What the transfer does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferSettings {
    /// Output side, texels (square tile).
    pub size: u32,
    /// The farthest scan point a texel may take, metres.
    pub max_distance: f32,
    /// Smallest cosine between the fitted normal and the scan triangle's.
    pub min_cos: f32,
    /// Strength `s` of the occlusion removal, `0..=1`.
    pub occlusion_strength: f32,
    /// Smallest divisor of the occlusion removal.
    pub occlusion_floor: f32,
    /// A speck is brighter than `speck_ratio` × the local mean luminance ...
    pub speck_ratio: f32,
    /// ... and its smallest channel is at least this fraction of its largest (nearly white).
    pub speck_whiteness: f32,
    /// Half side of the local-mean window, texels.
    pub speck_radius: u32,
    /// Padding passes outside the islands.
    pub padding: u32,
}

impl Default for TransferSettings {
    /// 2048², 1 cm reach, 60°, occlusion strength 0.7 with floor 0.5, specks above 1.35× the mean
    /// of a 17-texel window with min/max channel ≥ 0.6, 8 padding passes.
    fn default() -> Self {
        Self {
            size: 2048,
            max_distance: 0.01,
            min_cos: 0.5,
            occlusion_strength: 0.7,
            occlusion_floor: 0.5,
            speck_ratio: 1.35,
            speck_whiteness: 0.6,
            speck_radius: 8,
            padding: 8,
        }
    }
}

/// The fitted template surface of one transfer: nodes in the scan's frame with their triangles,
/// and per triangle corner the UV of the template vertex and per triangle its UV tile.
#[derive(Clone, Copy, Debug)]
pub struct FittedLayout<'a> {
    /// Fitted node positions (scan frame).
    pub nodes: &'a [Vec3],
    /// Three nodes per triangle.
    pub triangles: &'a [u32],
    /// The template UV of each triangle corner (tile-local, V down), parallel to `triangles`.
    pub corner_uvs: &'a [Vec2],
    /// The UV tile of each triangle.
    pub tiles: &'a [u16],
}

/// The scan side of a transfer.
#[derive(Clone, Copy, Debug)]
pub struct ScanSource<'a> {
    /// The scan (with UVs).
    pub scan: &'a ScanMesh,
    /// Its correspondence search.
    pub target: &'a TargetSurface,
    /// Its albedo.
    pub albedo: &'a Albedo,
    /// Ambient occlusion per scan vertex ([`scan_occlusion`]); empty: none removed.
    pub occlusion: &'a [f32],
}

/// One transferred tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferredAlbedo {
    /// Side, texels.
    pub size: u32,
    /// Linear Rec.709 albedo, rows top to bottom.
    pub texels: Vec<[f32; 3]>,
    /// Per texel a [`state`] value.
    pub states: Vec<u8>,
    /// Per texel the distance from the fitted surface to the scan, metres (negative: none).
    pub distance: Vec<f32>,
    /// Per texel the scan occlusion that was divided out (1 where nothing was).
    pub occlusion: Vec<f32>,
}

impl TransferredAlbedo {
    /// Counts of the texel states, indexed by [`state`] value.
    #[must_use]
    pub fn census(&self) -> [usize; 5] {
        let mut c = [0; 5];
        for &s in &self.states {
            if let Some(n) = c.get_mut(s as usize) {
                *n += 1;
            }
        }
        c
    }

    /// Fills empty texels next to filled ones with the mean of their filled 8-neighbours,
    /// `passes` rings outward.
    pub fn pad(&mut self, passes: u32) {
        let n = self.size as usize;
        let mut next = Vec::new();
        for _ in 0..passes {
            next.clear();
            for y in 0..n {
                for x in 0..n {
                    if self.states.get(y * n + x) != Some(&state::EMPTY) {
                        continue;
                    }
                    if let Some(c) = neighbour_mean(&self.texels, n, x, y, |i| {
                        self.states.get(i).is_some_and(|&s| s != state::EMPTY)
                    }) {
                        next.push((y * n + x, c));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            for &(i, c) in &next {
                if let (Some(t), Some(s)) = (self.texels.get_mut(i), self.states.get_mut(i)) {
                    *t = c;
                    *s = state::PADDING;
                }
            }
        }
    }
}

/// The mean of the 8-neighbours of `(x, y)` that pass `known`.
fn neighbour_mean(
    texels: &[[f32; 3]],
    n: usize,
    x: usize,
    y: usize,
    known: impl Fn(usize) -> bool,
) -> Option<[f32; 3]> {
    let (mut sum, mut count) = ([0.0_f32; 3], 0_u32);
    for dy in -1_i64..=1 {
        for dx in -1_i64..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (xx, yy) = (x as i64 + dx, y as i64 + dy);
            if xx < 0 || yy < 0 || xx >= n as i64 || yy >= n as i64 {
                continue;
            }
            let i = yy as usize * n + xx as usize;
            if known(i)
                && let Some(c) = texels.get(i)
            {
                for k in 0..3 {
                    sum[k] += c[k];
                }
                count += 1;
            }
        }
    }
    (count > 0).then(|| sum.map(|s| s / count as f32))
}

/// Ambient occlusion at every scan vertex: the cosine-weighted fraction of `rays` hemisphere rays
/// around the vertex normal that travel `max_distance` without hitting the scan (1 = open).
///
/// # Errors
/// [`CaptureError::Invalid`] for a ray count outside `1..=`[`MAX_OCCLUSION_RAYS`] or a
/// non-positive distance; [`CaptureError::Bake`] when the ray hierarchy cannot be built.
pub fn scan_occlusion(scan: &ScanMesh, rays: u32, max_distance: f32) -> Result<Vec<f32>> {
    if !((1..=MAX_OCCLUSION_RAYS).contains(&rays) && max_distance.is_finite() && max_distance > 0.0) {
        return Err(CaptureError::invalid("occlusion needs 1..=1024 rays and a positive distance"));
    }
    let bvh = Bvh::build(&scan.positions, &scan.indices).map_err(CaptureError::Bake)?;
    // Start rays a little off the surface: 0.2 mm, under the scan's millimetre noise.
    let offset = 2e-4;
    let mut out = Vec::with_capacity(scan.positions.len());
    for (v, (&p, &n)) in scan.positions.iter().zip(&scan.normals).enumerate() {
        if !(p.is_finite() && n.is_finite()) || n.length_squared() < 0.5 {
            out.push(1.0);
            continue;
        }
        let r = rotation(texel_hash(v as u32, 0, 0x5ca1));
        let origin = p + n * offset;
        let mut open = 0_u32;
        for i in 0..rays {
            let d = to_world(cosine_hemisphere(point(i, rays, r)), n);
            if !bvh.occluded(origin, d, 0.0, max_distance) {
                open += 1;
            }
        }
        out.push(open as f32 / rays as f32);
    }
    Ok(out)
}

/// Transfers the scan albedo into tile `tile` of the fitted layout (module documentation),
/// with occlusion removal; specks and padding are separate steps ([`find_specks`],
/// [`inpaint`], [`TransferredAlbedo::pad`]) run by [`transfer_and_clean`].
///
/// # Errors
/// [`CaptureError::Invalid`] for mismatched inputs, a scan without UVs or bad settings.
pub fn transfer(
    layout: &FittedLayout<'_>,
    source: &ScanSource<'_>,
    tile: u16,
    settings: &TransferSettings,
) -> Result<TransferredAlbedo> {
    let FittedLayout { nodes, triangles, corner_uvs, tiles } = *layout;
    if !triangles.len().is_multiple_of(3)
        || corner_uvs.len() != triangles.len()
        || tiles.len() != triangles.len() / 3
        || triangles.iter().any(|&i| i as usize >= nodes.len())
    {
        return Err(CaptureError::invalid("the fitted layout's arrays do not match"));
    }
    let scan = source.scan;
    if scan.uvs.len() != scan.positions.len() {
        return Err(CaptureError::invalid("the scan has no texture coordinates"));
    }
    if !source.occlusion.is_empty() && source.occlusion.len() != scan.positions.len() {
        return Err(CaptureError::invalid("the scan occlusion needs one value per scan vertex"));
    }
    if !(1..=MAX_TRANSFER_SIDE).contains(&settings.size) {
        return Err(CaptureError::invalid("the transfer size is outside 1..=8192"));
    }
    let n = settings.size as usize;
    let mut out = TransferredAlbedo {
        size: settings.size,
        texels: vec![[0.0; 3]; n * n],
        states: vec![state::EMPTY; n * n],
        distance: vec![-1.0; n * n],
        occlusion: vec![1.0; n * n],
    };
    let normals = node_normals(nodes, triangles);
    let size = settings.size as f32;
    for (t, tri) in triangles.as_chunks::<3>().0.iter().enumerate() {
        if tiles.get(t) != Some(&tile) {
            continue;
        }
        let uv = [0, 1, 2].map(|k| corner_uvs.get(3 * t + k).copied().unwrap_or(Vec2::ZERO) * size);
        let p = tri.map(|i| nodes.get(i as usize).copied().unwrap_or(Vec3::ZERO));
        let nn = tri.map(|i| normals.get(i as usize).copied().unwrap_or(Vec3::Y));
        let area = (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        let lo = uv[0].min(uv[1]).min(uv[2]);
        let hi = uv[0].max(uv[1]).max(uv[2]);
        let (x0, y0) = (lo.x.floor().max(0.0) as usize, lo.y.floor().max(0.0) as usize);
        let (x1, y1) = ((hi.x.ceil() as usize).min(n), (hi.y.ceil() as usize).min(n));
        for y in y0..y1 {
            for x in x0..x1 {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let b0 = (uv[1] - c).perp_dot(uv[2] - c) / area;
                let b1 = (uv[2] - c).perp_dot(uv[0] - c) / area;
                let b2 = 1.0 - b0 - b1;
                if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
                    continue;
                }
                let i = y * n + x;
                if out.states.get(i) != Some(&state::EMPTY) {
                    continue;
                }
                let q = p[0] * b0 + p[1] * b1 + p[2] * b2;
                let normal = (nn[0] * b0 + nn[1] * b1 + nn[2] * b2).normalize_or(Vec3::Y);
                sample_scan(source, q, normal, settings, &mut out, i);
            }
        }
    }
    Ok(out)
}

/// Fills texel `i` of `out` from the scan point closest to `q`.
fn sample_scan(
    source: &ScanSource<'_>,
    q: Vec3,
    normal: Vec3,
    settings: &TransferSettings,
    out: &mut TransferredAlbedo,
    i: usize,
) {
    let scan = source.scan;
    let hit = source.target.closest(q, normal, settings.min_cos, settings.max_distance);
    let corners = hit.and_then(|(_, _, t)| scan.triangle(t as usize));
    let (Some((_, distance, _)), Some(c)) = (hit, corners) else {
        if let Some(s) = out.states.get_mut(i) {
            *s = state::NO_SCAN;
        }
        return;
    };
    let tri = c.map(|v| scan.positions.get(v as usize).copied().unwrap_or(Vec3::ZERO));
    let (_, bary) = closest_on_triangle(q, &tri);
    let mut uv = Vec2::ZERO;
    let mut ao = 0.0;
    for (&v, &b) in c.iter().zip(&bary) {
        uv += scan.uvs.get(v as usize).copied().unwrap_or(Vec2::ZERO) * b;
        ao += source.occlusion.get(v as usize).copied().unwrap_or(1.0) * b;
    }
    let observed = source.albedo.sample(uv);
    let s = settings.occlusion_strength.clamp(0.0, 1.0);
    let divisor = (1.0 - s * (1.0 - ao.clamp(0.0, 1.0))).max(settings.occlusion_floor.max(1e-3));
    if let (Some(t), Some(st), Some(d), Some(o)) =
        (out.texels.get_mut(i), out.states.get_mut(i), out.distance.get_mut(i), out.occlusion.get_mut(i))
    {
        *t = observed.map(|x| (x / divisor).clamp(0.0, 1.0));
        *st = state::SCAN;
        *d = distance;
        *o = ao;
    }
}

/// Area-weighted unit normals of `nodes`.
fn node_normals(nodes: &[Vec3], triangles: &[u32]) -> Vec<Vec3> {
    let mut sum = vec![Vec3::ZERO; nodes.len()];
    for t in triangles.as_chunks::<3>().0 {
        let p = t.map(|i| nodes.get(i as usize).copied().unwrap_or(Vec3::ZERO));
        let f = (p[1] - p[0]).cross(p[2] - p[0]);
        for &i in t {
            if let Some(s) = sum.get_mut(i as usize) {
                *s += f;
            }
        }
    }
    sum.into_iter().map(|v| v.normalize_or(Vec3::Y)).collect()
}

/// Rec.709 luminance.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The scan texels of `albedo` that are specks: brighter than `ratio` × the mean luminance of the
/// scan texels in the `(2r + 1)²` window around them, and nearly white (smallest channel at
/// least `whiteness` × the largest). The mask is grown by one texel (a speck's rim is lighter too).
#[must_use]
pub fn find_specks(albedo: &TransferredAlbedo, ratio: f32, whiteness: f32, radius: u32) -> Vec<bool> {
    let n = albedo.size as usize;
    // Summed-area tables of luminance and count over the scan texels.
    let mut sum = vec![0.0_f64; (n + 1) * (n + 1)];
    let mut cnt = vec![0_u32; (n + 1) * (n + 1)];
    for y in 0..n {
        let (mut row_s, mut row_c) = (0.0_f64, 0_u32);
        for x in 0..n {
            let i = y * n + x;
            if albedo.states.get(i) == Some(&state::SCAN) {
                row_s += f64::from(luminance(albedo.texels.get(i).copied().unwrap_or([0.0; 3])));
                row_c += 1;
            }
            let (a, b) = ((y + 1) * (n + 1) + x + 1, y * (n + 1) + x + 1);
            let (above_s, above_c) = (sum.get(b).copied().unwrap_or(0.0), cnt.get(b).copied().unwrap_or(0));
            if let (Some(s), Some(c)) = (sum.get_mut(a), cnt.get_mut(a)) {
                *s = above_s + row_s;
                *c = above_c + row_c;
            }
        }
    }
    let r = radius as usize;
    let at = |x: usize, y: usize| {
        let k = y * (n + 1) + x;
        (sum.get(k).copied().unwrap_or(0.0), cnt.get(k).copied().unwrap_or(0))
    };
    let mut core = vec![false; n * n];
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            if albedo.states.get(i) != Some(&state::SCAN) {
                continue;
            }
            let c = albedo.texels.get(i).copied().unwrap_or([0.0; 3]);
            let (lo, hi) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
            if hi <= 0.0 || lo < whiteness * hi {
                continue;
            }
            let (xa, ya, xb, yb) = (x.saturating_sub(r), y.saturating_sub(r), (x + r + 1).min(n), (y + r + 1).min(n));
            let (s11, c11) = at(xb, yb);
            let (s01, c01) = at(xa, yb);
            let (s10, c10) = at(xb, ya);
            let (s00, c00) = at(xa, ya);
            let count = (c11 + c00).saturating_sub(c01 + c10);
            if count == 0 {
                continue;
            }
            let mean = (s11 - s01 - s10 + s00) / f64::from(count);
            if f64::from(luminance(c)) > f64::from(ratio) * mean {
                core[i] = true;
            }
        }
    }
    let mut mask = core.clone();
    for y in 0..n {
        for x in 0..n {
            if !core[y * n + x] {
                continue;
            }
            for (dx, dy) in [(-1_i64, 0_i64), (1, 0), (0, -1), (0, 1)] {
                let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                if xx >= 0 && yy >= 0 && (xx as usize) < n && (yy as usize) < n {
                    let j = yy as usize * n + xx as usize;
                    if albedo.states.get(j) == Some(&state::SCAN)
                        && let Some(m) = mask.get_mut(j)
                    {
                        *m = true;
                    }
                }
            }
        }
    }
    mask
}

/// Replaces the masked texels (and the covered texels without a scan point) by the mean of their
/// known neighbours, ring by ring from the outside in; at most `passes` rings.
pub fn inpaint(albedo: &mut TransferredAlbedo, mask: &[bool], passes: u32) {
    let n = albedo.size as usize;
    let mut todo: Vec<usize> = (0..n * n)
        .filter(|&i| mask.get(i).copied().unwrap_or(false) || albedo.states.get(i) == Some(&state::NO_SCAN))
        .collect();
    let mut known: Vec<bool> = (0..n * n)
        .map(|i| albedo.states.get(i) == Some(&state::SCAN) && !mask.get(i).copied().unwrap_or(false))
        .collect();
    for (i, s) in albedo.states.iter_mut().enumerate() {
        if mask.get(i).copied().unwrap_or(false) {
            *s = state::INPAINTED;
        }
    }
    let mut filled = Vec::new();
    for _ in 0..passes {
        filled.clear();
        for &i in &todo {
            if let Some(c) = neighbour_mean(&albedo.texels, n, i % n, i / n, |j| known.get(j).copied().unwrap_or(false))
            {
                filled.push((i, c));
            }
        }
        if filled.is_empty() {
            break;
        }
        for &(i, c) in &filled {
            if let (Some(t), Some(k)) = (albedo.texels.get_mut(i), known.get_mut(i)) {
                *t = c;
                *k = true;
            }
        }
        todo.retain(|&i| !known.get(i).copied().unwrap_or(true));
    }
}

/// [`transfer`], then speck removal, inpainting and padding as `settings` say. Returns the tile
/// and the number of speck texels found.
///
/// # Errors
/// As [`transfer`].
pub fn transfer_and_clean(
    layout: &FittedLayout<'_>,
    source: &ScanSource<'_>,
    tile: u16,
    settings: &TransferSettings,
) -> Result<(TransferredAlbedo, usize)> {
    let mut out = transfer(layout, source, tile, settings)?;
    let mask = find_specks(&out, settings.speck_ratio, settings.speck_whiteness, settings.speck_radius);
    let specks = mask.iter().filter(|&&m| m).count();
    inpaint(&mut out, &mask, 64);
    out.pad(settings.padding);
    Ok((out, specks))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat square scan in the z = 0 plane, 1 m wide, `k × k` quads, UVs = (x, 1 − y).
    fn plane_scan(k: usize) -> ScanMesh {
        let mut s = ScanMesh::default();
        for j in 0..=k {
            for i in 0..=k {
                let (x, y) = (i as f32 / k as f32, j as f32 / k as f32);
                s.positions.push(Vec3::new(x, y, 0.0));
                s.normals.push(Vec3::Z);
                s.uvs.push(Vec2::new(x, 1.0 - y));
                s.weld.push(s.weld.len() as u32);
            }
        }
        let at = |i: usize, j: usize| (j * (k + 1) + i) as u32;
        for j in 0..k {
            for i in 0..k {
                s.indices.extend_from_slice(&[at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                s.indices.extend_from_slice(&[at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        s
    }

    /// An albedo whose colour encodes the UV, with one white speck at `speck`.
    fn albedo(size: u32, speck: (u32, u32)) -> Albedo {
        let mut texels = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let (u, v) = ((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
                texels.push(if (x, y) == speck { [0.95, 0.95, 0.95] } else { [0.2 + 0.3 * u, 0.2 + 0.3 * v, 0.3] });
            }
        }
        Albedo { width: size, height: size, texels }
    }

    /// A fitted layout: the square [0.25, 0.75]² 1 mm above the plane as two triangles whose UVs
    /// fill tile 0, plus one triangle in tile 1.
    fn layout() -> (Vec<Vec3>, Vec<u32>, Vec<Vec2>, Vec<u16>) {
        let nodes = vec![
            Vec3::new(0.25, 0.25, 0.001),
            Vec3::new(0.75, 0.25, 0.001),
            Vec3::new(0.75, 0.75, 0.001),
            Vec3::new(0.25, 0.75, 0.001),
        ];
        let tris = vec![0, 1, 2, 0, 2, 3, 0, 1, 2];
        let uvs = vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        ];
        (nodes, tris, uvs, vec![0, 0, 1])
    }

    #[test]
    fn wp25_albedo_transfers_through_the_fit_and_specks_are_inpainted() {
        let scan = plane_scan(16);
        let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
        // The speck sits at scan UV (0.5, 0.5): the middle of the fitted square.
        let albedo = albedo(64, (32, 32));
        let (nodes, triangles, corner_uvs, tiles) = layout();
        let lay = FittedLayout { nodes: &nodes, triangles: &triangles, corner_uvs: &corner_uvs, tiles: &tiles };
        let source = ScanSource { scan: &scan, target: &target, albedo: &albedo, occlusion: &[] };
        let settings = TransferSettings { size: 32, speck_radius: 4, ..TransferSettings::default() };
        let raw = transfer(&lay, &source, 0, &settings).unwrap();
        // Every texel of tile 0 is covered; tile UV (u, v) lies at scan point
        // (0.25 + 0.5 u, 0.25 + 0.5 (1 − v)), i.e. scan UV (0.25 + 0.5 u, 0.25 + 0.5 v).
        assert_eq!(raw.census()[usize::from(state::SCAN)], 32 * 32);
        for &(x, y) in &[(3_usize, 5_usize), (20, 9), (30, 30)] {
            let (u, v) = ((x as f32 + 0.5) / 32.0, (y as f32 + 0.5) / 32.0);
            let (su, sv) = (0.25 + 0.5 * u, 0.25 + 0.5 * v);
            let got = raw.texels[y * 32 + x];
            assert!((got[0] - (0.2 + 0.3 * su)).abs() < 0.01 && (got[1] - (0.2 + 0.3 * sv)).abs() < 0.01, "{got:?}");
            assert!((raw.distance[y * 32 + x] - 0.001).abs() < 1e-5);
        }
        let (clean, specks) = transfer_and_clean(&lay, &source, 0, &settings).unwrap();
        assert!((1..=40).contains(&specks), "{specks} speck texels");
        let max_lum = clean.texels.iter().map(|&c| luminance(c)).fold(0.0, f32::max);
        assert!(max_lum < 0.5, "a speck survived: {max_lum}");
        assert_eq!(clean.census()[usize::from(state::INPAINTED)], specks);
        // Tile 1 holds only the third triangle: the rest is empty until padded.
        let other = transfer(&lay, &source, 1, &settings).unwrap();
        let c = other.census();
        assert!(c[usize::from(state::SCAN)] > 400 && c[usize::from(state::EMPTY)] > 400, "{c:?}");
    }

    #[test]
    fn wp25_occlusion_is_divided_out_and_measured_on_the_scan() {
        // Open plane: every vertex sees the whole hemisphere.
        let scan = plane_scan(4);
        let ao = scan_occlusion(&scan, 64, 0.5).unwrap();
        assert!(ao.iter().all(|&a| a == 1.0));
        // A wall standing on the plane at x = 0.5 hides a quarter of the hemisphere's
        // cosine-weighted light from the vertex at its foot ... about half of it.
        let mut walled = plane_scan(4);
        let wall = plane_scan(1);
        let mut w = wall.clone();
        for p in &mut w.positions {
            *p = Vec3::new(0.5 + 1e-3, p.x, p.y);
        }
        for n in &mut w.normals {
            *n = -Vec3::X;
        }
        walled.append(&w).unwrap();
        let ao = scan_occlusion(&walled, 256, 0.5).unwrap();
        // Vertex (x = 0.25, y = 0.5): index 2 * 5 + 1.
        let foot = ao[2 * 5 + 1];
        assert!(foot < 0.95 && foot > 0.5, "{foot}");
        // With occlusion 0.5 everywhere, strength 1 and floor 0.25, the albedo doubles.
        let target = TargetSurface::new(&scan, vec![1.0; scan.triangle_count()]).unwrap();
        let albedo = albedo(16, (99, 99));
        let occlusion = vec![0.5; scan.positions.len()];
        let (nodes, triangles, corner_uvs, tiles) = layout();
        let lay = FittedLayout { nodes: &nodes, triangles: &triangles, corner_uvs: &corner_uvs, tiles: &tiles };
        let source = ScanSource { scan: &scan, target: &target, albedo: &albedo, occlusion: &occlusion };
        let s =
            TransferSettings { size: 8, occlusion_strength: 1.0, occlusion_floor: 0.25, ..TransferSettings::default() };
        let plain = transfer(&lay, &ScanSource { occlusion: &[], ..source }, 0, &s).unwrap();
        let lit = transfer(&lay, &source, 0, &s).unwrap();
        for (a, b) in plain.texels.iter().zip(&lit.texels) {
            assert!((b[0] - 2.0 * a[0]).abs() < 1e-4);
        }
        assert!(scan_occlusion(&scan, 0, 0.5).is_err());
        assert!(transfer(&lay, &source, 0, &TransferSettings { size: 0, ..s }).is_err());
    }
}
