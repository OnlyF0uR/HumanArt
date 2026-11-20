//! The eye's frame on a deformed mesh: eyeball centre, radius and gaze axis fitted to the
//! eyeball part's vertices, wherever the rig has moved them.
//!
//! [`EyeVertices::find`] splits the eyeball part once into scleral and corneal vertices (by their
//! distance from a sphere fitted to the rest shape); [`EyeVertices::frame`] fits a sphere to the
//! deformed scleral vertices (linear least squares, allocation-free) and takes the axis toward the
//! corneal vertices' mean. The penetration guard and the eye shader inputs use it.

use humanart_math::{DVec3, Vec3};

use crate::error::{ModelError, Result};
use crate::template::Template;

/// The fitted frame of one eyeball.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeFrame {
    /// Eyeball centre.
    pub centre: Vec3,
    /// Scleral sphere radius.
    pub radius: f32,
    /// Unit gaze axis (centre toward the cornea).
    pub axis: Vec3,
}

/// Least-squares sphere through the points `positions[i]` for `i` in `indices`: (centre,
/// radius); `None` for fewer than 4 points or a degenerate set.
#[must_use]
pub fn fit_sphere(positions: &[Vec3], indices: &[u32]) -> Option<(Vec3, f32)> {
    let mut mean = DVec3::ZERO;
    let mut n = 0.0_f64;
    for &i in indices {
        mean += positions.get(i as usize)?.as_dvec3();
        n += 1.0;
    }
    if n < 4.0 {
        return None;
    }
    mean /= n;
    // Normal equations of 2 p·c + k = |p|² (p relative to the mean).
    let mut a = [[0.0_f64; 5]; 4];
    for &i in indices {
        let p = positions.get(i as usize)?.as_dvec3() - mean;
        let row = [2.0 * p.x, 2.0 * p.y, 2.0 * p.z, 1.0];
        let b = p.length_squared();
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] += row[r] * row[c];
            }
            a[r][4] += row[r] * b;
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        a.swap(col, pivot);
        let d = a[col][col];
        if d.abs() < 1e-30 {
            return None;
        }
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / d;
                for (x, &y) in row.iter_mut().zip(&pivot_row).skip(col) {
                    *x -= f * y;
                }
            }
        }
    }
    let c = DVec3::new(a[0][4] / a[0][0], a[1][4] / a[1][1], a[2][4] / a[2][2]);
    let k = a[3][4] / a[3][3];
    let r2 = k + c.length_squared();
    if !(r2 > 0.0 && r2.is_finite()) {
        return None;
    }
    let centre = mean + c;
    Some((Vec3::new(centre.x as f32, centre.y as f32, centre.z as f32), r2.sqrt() as f32))
}

/// The scleral and corneal vertices of one eyeball part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EyeVertices {
    /// Vertices on the scleral sphere.
    pub sclera: Vec<u32>,
    /// Vertices on the corneal bulge.
    pub cornea: Vec<u32>,
}

impl EyeVertices {
    /// Splits the part `part` (`eyeball-L` or `eyeball-R`) of the template's mesh.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when the part does not exist or is not a sphere with a bulge.
    pub fn find(template: &Template, part: &str) -> Result<Self> {
        let mesh = template.mesh();
        let p = mesh.parts().iter().find(|p| p.name == part).ok_or(ModelError::invalid(
            "eyeball part",
            0,
            "no part with that name",
        ))?;
        let mut used = vec![false; mesh.vertex_count()];
        for f in p.faces() {
            for &v in mesh.face(f) {
                if let Some(u) = used.get_mut(v as usize) {
                    *u = true;
                }
            }
        }
        let all: Vec<u32> = (0..used.len() as u32).filter(|&v| used[v as usize]).collect();
        let positions = mesh.positions();
        let (c, _) = fit_sphere(positions, &all).ok_or(ModelError::invalid("eyeball part", 1, "no sphere fits"))?;
        // The sclera: the vertices at the most common distance (the median), refined once.
        let mut d: Vec<f32> = all.iter().map(|&v| positions[v as usize].distance(c)).collect();
        d.sort_by(f32::total_cmp);
        let median = d.get(d.len() / 2).copied().unwrap_or(0.0);
        let near: Vec<u32> =
            all.iter().copied().filter(|&v| (positions[v as usize].distance(c) - median).abs() < 1e-4).collect();
        let (c, r) = fit_sphere(positions, &near).ok_or(ModelError::invalid("eyeball part", 2, "no sphere fits"))?;
        let (mut sclera, mut cornea) = (Vec::new(), Vec::new());
        for &v in &all {
            let dist = positions[v as usize].distance(c);
            if dist > r + 2e-4 {
                cornea.push(v);
            } else if (dist - r).abs() < 1e-4 {
                sclera.push(v);
            }
        }
        if sclera.len() < 8 || cornea.is_empty() {
            return Err(ModelError::invalid("eyeball part", 3, "no corneal bulge or too few scleral vertices"));
        }
        Ok(Self { sclera, cornea })
    }

    /// The frame on (deformed) `positions`; `None` when the vertices are out of range or
    /// degenerate. No allocation.
    #[must_use]
    pub fn frame(&self, positions: &[Vec3]) -> Option<EyeFrame> {
        let (centre, radius) = fit_sphere(positions, &self.sclera)?;
        let mut sum = Vec3::ZERO;
        for &v in &self.cornea {
            sum += *positions.get(v as usize)?;
        }
        let axis = (sum / self.cornea.len().max(1) as f32 - centre).try_normalize()?;
        Some(EyeFrame { centre, radius, axis })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_fit_recovers_centre_and_radius() {
        let c = Vec3::new(0.03, 1.6, 0.07);
        let mut pts = Vec::new();
        for i in 0..10 {
            for j in 0..6 {
                let (a, b) = (i as f32 * 0.6, j as f32 * 0.5 - 1.2);
                pts.push(c + Vec3::new(b.cos() * a.cos(), b.sin(), b.cos() * a.sin()) * 0.012);
            }
        }
        let idx: Vec<u32> = (0..pts.len() as u32).collect();
        let (fc, r) = fit_sphere(&pts, &idx).unwrap();
        assert!(fc.distance(c) < 1e-6 && (r - 0.012).abs() < 1e-6, "{fc:?} {r}");
        assert!(fit_sphere(&pts, &idx[..3]).is_none());
        assert!(fit_sphere(&pts, &[0, 1, 2, 999]).is_none());
        // Coplanar points have no unique sphere.
        let flat: Vec<Vec3> = (0..8).map(|k| Vec3::new(k as f32, (k * k) as f32, 0.0)).collect();
        let fi: Vec<u32> = (0..8).collect();
        assert!(fit_sphere(&flat, &fi).is_none_or(|s| !s.1.is_finite() || s.1 > 1e3));
    }
}
