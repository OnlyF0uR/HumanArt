//! Low-frequency skin tone variation (todo 1.4): the slow, uneven change of pigment and blood over
//! a face or a limb (a slightly ruddier patch here, a paler one there) at the scale of centimetres,
//! on top of the zones' regional modulation (`super::zones`: redder cheeks and nose, darker
//! under-eyes) and without any grain.
//!
//! A value noise in three dimensions over the template's **rest** positions (so it is continuous
//! across UV seams and does not swim when the face moves), two octaves with lattice cells of
//! [`CELL_COARSE_M`] and [`CELL_FINE_M`], is computed per vertex ([`vertex_variation`]: 0.5 is
//! neutral, the range is 0..1) and rasterised into the compositor's baked input layer
//! ([`bake_tile_values`], the blue channel of `super::baked`). The compositor adds
//! `2 (v − 0.5)` times the zone table's `variation_melanin` and `variation_haemoglobin` to those
//! attributes. The cell sizes and the amplitudes the engine sets are **artistic** values (a look
//! checked on the stills), not measurements.

use humanart_math::{Vec2, Vec3};

use super::SkinError;
use super::noise::{fade, pcg, unit};
use crate::mesh::{Mesh, Triangulation};

/// Lattice cell of the coarse octave, metres.
pub const CELL_COARSE_M: f32 = 0.045;
/// Lattice cell of the fine octave, metres.
pub const CELL_FINE_M: f32 = 0.018;
/// Weight of the fine octave relative to the coarse one.
pub const FINE_WEIGHT: f32 = 0.5;
/// Largest tile edge [`bake_tile_values`] takes.
pub const MAX_VARIATION_SIZE: u32 = 4096;

/// Hash of a 3D lattice point and a seed.
fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    // Bit patterns of the signed lattice coordinates; the hash mixes all 32 bits.
    #[allow(clippy::cast_sign_loss)]
    let (x, y, z) = (x as u32, y as u32, z as u32);
    unit(pcg(x ^ pcg(y ^ pcg(z ^ pcg(seed)))))
}

/// Value noise in `[0, 1)` at `p` (lattice units).
fn value_noise_3d(p: Vec3, seed: u32) -> f32 {
    let f = Vec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let t = p - f;
    let (u, v, w) = (fade(t.x), fade(t.y), fade(t.z));
    // Lattice coordinates of skin positions in metres over centimetre cells are small integers.
    #[allow(clippy::cast_possible_truncation)]
    let (x, y, z) = (f.x as i32, f.y as i32, f.z as i32);
    let lerp = |a: f32, b: f32, s: f32| a + (b - a) * s;
    let c = |dx: i32, dy: i32, dz: i32| hash(x + dx, y + dy, z + dz, seed);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// The variation of every vertex at its rest position (metres), 0..1, 0.5 neutral.
pub fn vertex_variation(positions: &[Vec3], seed: u32) -> Vec<f32> {
    positions
        .iter()
        .map(|&p| {
            if !p.is_finite() {
                return 0.5;
            }
            let coarse = value_noise_3d(p / CELL_COARSE_M, seed) - 0.5;
            let fine = value_noise_3d(p / CELL_FINE_M, seed ^ 0x9e37_79b9) - 0.5;
            (0.5 + (coarse + FINE_WEIGHT * fine) / (1.0 + FINE_WEIGHT)).clamp(0.0, 1.0)
        })
        .collect()
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Rasterises one value per vertex into the `size²` texels of UV tile `tile` (texel `(x, y)`
/// samples UV `((x + 0.5) / size, (y + 0.5) / size)`), barycentrically; texels no triangle covers
/// take the value of a covered neighbour up to `dilate` texels away, the rest 0.5 (neutral).
///
/// # Errors
/// [`SkinError::Limit`] for a size above [`MAX_VARIATION_SIZE`]; [`SkinError::Invalid`] when
/// `values` or the triangles do not fit the mesh.
pub fn bake_tile_values(
    mesh: &Mesh,
    triangles: &Triangulation,
    values: &[f32],
    tile: u16,
    size: u32,
    dilate: u32,
) -> Result<Vec<f32>, SkinError> {
    if size == 0 || size > MAX_VARIATION_SIZE {
        return Err(SkinError::limit("variation tile size", u64::from(MAX_VARIATION_SIZE), u64::from(size)));
    }
    let uvs = mesh.uvs();
    if values.len() != uvs.len() || triangles.indices.len() != 3 * triangles.triangle_faces.len() {
        return Err(SkinError::Invalid { what: "variation values and triangles must match the mesh" });
    }
    let s = size as usize;
    let mut out = vec![f32::NAN; s * s];
    let scale = size as f32;
    let face_tiles = mesh.face_tiles();
    for t in 0..triangles.len() {
        let (Some(c), Some(&face)) = (triangles.triangle(t), triangles.triangle_faces.get(t)) else { continue };
        if face_tiles.get(face as usize) != Some(&tile) {
            continue;
        }
        let mut p = [Vec2::ZERO; 3];
        let mut v = [0.0_f32; 3];
        for k in 0..3 {
            let i = c[k] as usize;
            let (Some(uv), Some(&value)) = (uvs.get(i), values.get(i)) else {
                return Err(SkinError::Invalid { what: "a triangle corner is not a vertex" });
            };
            p[k] = *uv * scale;
            v[k] = value;
        }
        let area = edge(p[0], p[1], p[2]);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Clamped to the tile.
        let bound = |a: f32| a.clamp(0.0, scale - 1.0) as usize;
        let (x0, x1) =
            (bound(p[0].x.min(p[1].x).min(p[2].x) - 0.5), bound((p[0].x.max(p[1].x).max(p[2].x) - 0.5).ceil()));
        let (y0, y1) =
            (bound(p[0].y.min(p[1].y).min(p[2].y) - 0.5), bound((p[0].y.max(p[1].y).max(p[2].y) - 0.5).ceil()));
        let tol = -1e-4 * area.abs();
        let inv = 1.0 / area.abs();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let b0 = edge(p[1], p[2], q) * area.signum();
                let b1 = edge(p[2], p[0], q) * area.signum();
                let b2 = edge(p[0], p[1], q) * area.signum();
                if b0 < tol || b1 < tol || b2 < tol {
                    continue;
                }
                out[y * s + x] = (b0 * v[0] + b1 * v[1] + b2 * v[2]) * inv;
            }
        }
    }
    for _ in 0..dilate {
        let prev = out.clone();
        for y in 0..s {
            for x in 0..s {
                if !prev[y * s + x].is_nan() {
                    continue;
                }
                let n = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
                if let Some(v) =
                    n.iter().filter(|(a, b)| *a < s && *b < s).map(|(a, b)| prev[b * s + a]).find(|v| !v.is_nan())
                {
                    out[y * s + x] = v;
                }
            }
        }
    }
    for v in &mut out {
        if v.is_nan() {
            *v = 0.5;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_1_4_variation_is_smooth_neutral_on_average_and_continuous() {
        // Points 1 mm apart differ little (no grain); over a face-sized grid the mean is near 0.5
        // and the values spread.
        let mut values = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                values.push(Vec3::new(i as f32 * 0.005, j as f32 * 0.005, 0.1));
            }
        }
        let v = vertex_variation(&values, 3);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let (lo, hi) = v.iter().fold((1.0_f32, 0.0_f32), |(l, h), x| (l.min(*x), h.max(*x)));
        assert!((mean - 0.5).abs() < 0.1, "{mean}");
        assert!(hi - lo > 0.2, "it varies: {lo}..{hi}");
        let near = vertex_variation(&[Vec3::new(0.05, 0.05, 0.1), Vec3::new(0.051, 0.05, 0.1)], 3);
        assert!((near[0] - near[1]).abs() < 0.02, "no grain: {near:?}");
        assert_eq!(vertex_variation(&[Vec3::splat(f32::NAN)], 3), vec![0.5]);
        assert_eq!(vertex_variation(&values, 3), v, "deterministic");
    }
}
