//! The lid aperture `[EYE-05]`: where the eyelids open over the eyeball, as the eye shader's
//! occlusion and lid-shadow input.
//!
//! The lid margins (the template's `L.eye.upper.NN`, `L.eye.lower.NN` and canthus landmarks on the
//! deformed mesh) are converted to angles about the eyeball centre in the head frame (azimuth
//! `atan2(x, z)`, elevation `atan2(y, |xz|)`) and resampled at [`APERTURE_SAMPLES`] evenly spaced
//! azimuths between the two canthi. The shader interpolates the two curves and turns the angular
//! distance of an eyeball point from the nearer lid into ambient occlusion and a soft lid and
//! lash shadow (`shaders/eye/eye.wgsl`, [`LidAperture::openness`] is the CPU mirror). The aperture
//! follows blinks and squints because it is recomputed from the deformed landmarks.

use humanart_math::{Quat, Vec3};

use crate::error::{ModelError, Result};
use crate::landmark::Landmark;

/// Samples per lid curve.
pub const APERTURE_SAMPLES: usize = 8;

/// The most margin landmarks per lid.
pub const MAX_MARGIN_POINTS: usize = 32;

/// The opening of one eye's lids, in angles about the eyeball centre (head frame), radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidAperture {
    /// Azimuth of the first sample (the canthus with the smaller azimuth).
    pub from: f32,
    /// Azimuth of the last sample.
    pub to: f32,
    /// Elevation of the upper lid margin at each sample.
    pub upper: [f32; APERTURE_SAMPLES],
    /// Elevation of the lower lid margin at each sample (never above the upper).
    pub lower: [f32; APERTURE_SAMPLES],
}

impl Default for LidAperture {
    /// A wide-open almond about the axis (±0.5 rad across, +0.3 / −0.25 rad high).
    fn default() -> Self {
        let mut upper = [0.0; APERTURE_SAMPLES];
        let mut lower = [0.0; APERTURE_SAMPLES];
        for k in 0..APERTURE_SAMPLES {
            let s = (std::f32::consts::PI * k as f32 / (APERTURE_SAMPLES - 1) as f32).sin();
            upper[k] = 0.3 * s;
            lower[k] = -0.25 * s;
        }
        Self { from: -0.5, to: 0.5, upper, lower }
    }
}

/// Azimuth and elevation of `p` about `centre` in the head frame `head`.
#[must_use]
pub fn angles(centre: Vec3, head: Quat, p: Vec3) -> (f32, f32) {
    let d = head.conjugate().mul_vec3(p - centre);
    (d.x.atan2(d.z), d.y.atan2((d.x * d.x + d.z * d.z).sqrt()))
}

/// Linear interpolation of the curve through `points` (sorted by azimuth) at azimuth `a`,
/// constant beyond the ends.
fn interpolate(points: &[(f32, f32)], a: f32) -> f32 {
    let Some(&first) = points.first() else { return 0.0 };
    if a <= first.0 {
        return first.1;
    }
    for w in points.windows(2) {
        let (p, q) = (w[0], w[1]);
        if a <= q.0 {
            let span = q.0 - p.0;
            return if span > 1e-9 { p.1 + (q.1 - p.1) * (a - p.0) / span } else { q.1 };
        }
    }
    points.last().map_or(first.1, |l| l.1)
}

impl LidAperture {
    /// The aperture of margin points `upper` and `lower` (any order) and the two canthi, about
    /// `centre` in the head frame `head`.
    ///
    /// # Errors
    /// [`ModelError::Limit`] for more than [`MAX_MARGIN_POINTS`] points per lid;
    /// [`ModelError::Invalid`] for non-finite points or coincident canthi.
    pub fn from_margin(centre: Vec3, head: Quat, upper: &[Vec3], lower: &[Vec3], canthi: (Vec3, Vec3)) -> Result<Self> {
        for lid in [upper, lower] {
            if lid.len() > MAX_MARGIN_POINTS {
                return Err(ModelError::limit("lid margin points", MAX_MARGIN_POINTS as u64, lid.len() as u64));
            }
        }
        let (a0, a1) = (angles(centre, head, canthi.0), angles(centre, head, canthi.1));
        let (first, last) = if a0.0 <= a1.0 { (a0, a1) } else { (a1, a0) };
        if !(first.0.is_finite() && last.0.is_finite() && first.1.is_finite() && last.1.is_finite())
            || last.0 - first.0 < 1e-4
        {
            return Err(ModelError::invalid("lid aperture", 0, "canthi coincide or are not finite"));
        }
        let mut out =
            Self { from: first.0, to: last.0, upper: [0.0; APERTURE_SAMPLES], lower: [0.0; APERTURE_SAMPLES] };
        let mut buffer = [(0.0_f32, 0.0_f32); MAX_MARGIN_POINTS + 2];
        for (lid, samples) in [(upper, &mut out.upper), (lower, &mut out.lower)] {
            let n = lid.len() + 2;
            buffer[0] = first;
            buffer[1] = last;
            for (slot, &p) in buffer[2..n].iter_mut().zip(lid) {
                *slot = angles(centre, head, p);
                if !(slot.0.is_finite() && slot.1.is_finite()) {
                    return Err(ModelError::invalid("lid aperture", 1, "a margin point is not finite"));
                }
            }
            let points = &mut buffer[..n];
            points.sort_by(|p, q| p.0.total_cmp(&q.0));
            for (k, s) in samples.iter_mut().enumerate() {
                let a = out.from + (out.to - out.from) * k as f32 / (APERTURE_SAMPLES - 1) as f32;
                *s = interpolate(points, a);
            }
        }
        // Lids that meet or cross (a closed eye) close the aperture at their mean.
        for k in 0..APERTURE_SAMPLES {
            if out.upper[k] < out.lower[k] {
                let m = f32::midpoint(out.upper[k], out.lower[k]);
                out.upper[k] = m;
                out.lower[k] = m;
            }
        }
        Ok(out)
    }

    /// The upper and lower elevations at azimuth `a` (clamped to the canthi).
    #[must_use]
    pub fn lids_at(&self, a: f32) -> (f32, f32) {
        let span = (self.to - self.from).max(1e-6);
        let x = ((a - self.from) / span).clamp(0.0, 1.0) * (APERTURE_SAMPLES - 1) as f32;
        let k = (x.floor() as usize).min(APERTURE_SAMPLES - 2);
        let f = x - k as f32;
        let lerp = |c: &[f32; APERTURE_SAMPLES]| c[k] + (c[k + 1] - c[k]) * f;
        (lerp(&self.upper), lerp(&self.lower))
    }

    /// Signed angular distance (radians) of the direction `(azimuth, elevation)` inside the
    /// opening: positive inside, the distance to the nearer lid; negative under a lid or beyond a
    /// canthus. The shader's occlusion input.
    #[must_use]
    pub fn openness(&self, azimuth: f32, elevation: f32) -> f32 {
        let (up, low) = self.lids_at(azimuth);
        let vertical = (up - elevation).min(elevation - low);
        let horizontal = (azimuth - self.from).min(self.to - azimuth);
        vertical.min(horizontal)
    }

    /// Height of the opening (upper minus lower elevation) at its widest sample, radians.
    #[must_use]
    pub fn max_height(&self) -> f32 {
        self.upper.iter().zip(&self.lower).map(|(u, l)| u - l).fold(0.0, f32::max)
    }
}

/// The landmark indices of one eye's lid margins, found once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LidLandmarks {
    /// Upper margin landmarks.
    pub upper: Vec<usize>,
    /// Lower margin landmarks.
    pub lower: Vec<usize>,
    /// Medial and lateral canthus.
    pub canthi: (usize, usize),
}

impl LidLandmarks {
    /// Finds the landmarks `{side}.eye.upper.NN`, `{side}.eye.lower.NN`,
    /// `{side}.eye.canthus.medial` and `.lateral` (`side` is `"L"` or `"R"`).
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when a lid has no margin landmark or a canthus is missing;
    /// [`ModelError::Limit`] beyond [`MAX_MARGIN_POINTS`].
    pub fn find(landmarks: &[Landmark], side: &str) -> Result<Self> {
        let upper_prefix = format!("{side}.eye.upper.");
        let lower_prefix = format!("{side}.eye.lower.");
        let mut upper = Vec::new();
        let mut lower = Vec::new();
        let (mut medial, mut lateral) = (None, None);
        let medial_name = format!("{side}.eye.canthus.medial");
        let lateral_name = format!("{side}.eye.canthus.lateral");
        for (i, l) in landmarks.iter().enumerate() {
            if l.name.starts_with(&upper_prefix) {
                upper.push(i);
            } else if l.name.starts_with(&lower_prefix) {
                lower.push(i);
            } else if l.name == medial_name {
                medial = Some(i);
            } else if l.name == lateral_name {
                lateral = Some(i);
            }
        }
        if upper.len().max(lower.len()) > MAX_MARGIN_POINTS {
            return Err(ModelError::limit(
                "lid margin landmarks",
                MAX_MARGIN_POINTS as u64,
                upper.len().max(lower.len()) as u64,
            ));
        }
        let (Some(medial), Some(lateral)) = (medial, lateral) else {
            return Err(ModelError::invalid("lid landmarks", 0, "a canthus landmark is missing"));
        };
        if upper.is_empty() || lower.is_empty() {
            return Err(ModelError::invalid("lid landmarks", 1, "a lid has no margin landmark"));
        }
        Ok(Self { upper, lower, canthi: (medial, lateral) })
    }

    /// The aperture on deformed `positions`, about `centre` in the head frame `head`. No
    /// allocation.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when a landmark's vertex is out of range, and the errors of
    /// [`LidAperture::from_margin`].
    pub fn aperture(
        &self,
        landmarks: &[Landmark],
        positions: &[Vec3],
        centre: Vec3,
        head: Quat,
    ) -> Result<LidAperture> {
        let at = |i: usize| -> Result<Vec3> {
            landmarks.get(i).and_then(|l| l.position(positions)).ok_or(ModelError::invalid(
                "lid landmarks",
                i,
                "landmark vertex out of range",
            ))
        };
        let mut up = [Vec3::ZERO; MAX_MARGIN_POINTS];
        let mut low = [Vec3::ZERO; MAX_MARGIN_POINTS];
        for (slot, &i) in up.iter_mut().zip(&self.upper) {
            *slot = at(i)?;
        }
        for (slot, &i) in low.iter_mut().zip(&self.lower) {
            *slot = at(i)?;
        }
        LidAperture::from_margin(
            centre,
            head,
            &up[..self.upper.len()],
            &low[..self.lower.len()],
            (at(self.canthi.0)?, at(self.canthi.1)?),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_sphere(az: f32, el: f32) -> Vec3 {
        Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos()) * 0.012
    }

    #[test]
    fn eye_05_aperture_resamples_the_margins() {
        let upper: Vec<Vec3> = (1..6).map(|k| k as f32 * 0.15 - 0.45).map(|a| on_sphere(a, 0.3 - a * a)).collect();
        let lower: Vec<Vec3> = (1..6).map(|k| k as f32 * 0.15 - 0.45).map(|a| on_sphere(a, -0.2 + a * a)).collect();
        let ap = LidAperture::from_margin(
            Vec3::ZERO,
            Quat::IDENTITY,
            &upper,
            &lower,
            (on_sphere(0.5, 0.02), on_sphere(-0.5, 0.0)),
        )
        .unwrap();
        assert!((ap.from + 0.5).abs() < 1e-5 && (ap.to - 0.5).abs() < 1e-5);
        // At the centre the curves pass through the margin points.
        let (u, l) = ap.lids_at(0.0);
        assert!((u - 0.3).abs() < 0.03 && (l + 0.2).abs() < 0.03, "{u} {l}");
        // Inside positive, under a lid or past a canthus negative.
        assert!(ap.openness(0.0, 0.0) > 0.15);
        assert!(ap.openness(0.0, 0.4) < 0.0);
        assert!(ap.openness(0.0, -0.3) < 0.0);
        assert!(ap.openness(0.7, 0.0) < 0.0);
        assert!(ap.max_height() > 0.45);
    }

    #[test]
    fn eye_05_crossed_lids_close_the_aperture() {
        let upper = [on_sphere(0.0, -0.05)];
        let lower = [on_sphere(0.0, 0.05)];
        let ap = LidAperture::from_margin(
            Vec3::ZERO,
            Quat::IDENTITY,
            &upper,
            &lower,
            (on_sphere(-0.4, 0.0), on_sphere(0.4, 0.0)),
        )
        .unwrap();
        assert!(ap.max_height() < 1e-6);
        assert!(ap.openness(0.0, 0.0) <= 1e-6);
    }

    #[test]
    fn bad_margins_are_refused() {
        let p = [Vec3::new(f32::NAN, 0.0, 1.0)];
        let ok = [on_sphere(0.0, 0.1)];
        let c = (on_sphere(-0.4, 0.0), on_sphere(0.4, 0.0));
        assert!(LidAperture::from_margin(Vec3::ZERO, Quat::IDENTITY, &p, &ok, c).is_err());
        assert!(LidAperture::from_margin(Vec3::ZERO, Quat::IDENTITY, &ok, &ok, (c.0, c.0)).is_err());
        let many = vec![on_sphere(0.0, 0.1); MAX_MARGIN_POINTS + 1];
        assert!(LidAperture::from_margin(Vec3::ZERO, Quat::IDENTITY, &many, &ok, c).is_err());
        assert!(LidLandmarks::find(&[], "L").is_err());
    }
}
