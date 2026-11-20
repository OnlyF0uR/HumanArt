//! Rigid + scale alignment from landmark pairs `[WP-25]`: the similarity transform (rotation,
//! translation, uniform scale) that best maps the template's landmarks onto a scan's, by weighted
//! Procrustes with Umeyama's reflection-safe scale (`humanart_math::solve::svd::procrustes`).

use humanart_math::Vec3;
use humanart_math::solve::svd::{kabsch, procrustes};

use super::{CaptureError, Result};

pub use humanart_model::scanfit::Similarity;

/// What [`fit_similarity`] found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alignment {
    /// The transform from source to target.
    pub transform: Similarity,
    /// The weighted RMS distance of the mapped source points from their targets, metres.
    pub rms: f32,
}

/// The similarity (or, with `with_scale` false, rigid) transform that best maps `source[i]` onto
/// `target[i]`, weighted by `weights` (all 1 when `None`).
///
/// # Errors
/// [`CaptureError::Invalid`] for fewer than 3 pairs or differing lengths; [`CaptureError::Solve`]
/// for degenerate input (coincident points, non-finite values, zero weights).
pub fn fit_similarity(
    source: &[Vec3],
    target: &[Vec3],
    weights: Option<&[f64]>,
    with_scale: bool,
) -> Result<Alignment> {
    if source.len() != target.len() || source.len() < 3 {
        return Err(CaptureError::invalid("an alignment needs at least 3 landmark pairs"));
    }
    let to64 = |v: &Vec3| [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
    let s: Vec<[f64; 3]> = source.iter().map(to64).collect();
    let t: Vec<[f64; 3]> = target.iter().map(to64).collect();
    let fit = if with_scale { procrustes(&s, &t, weights)? } else { kabsch(&s, &t, weights)? };
    let row = |r: [f64; 3]| Vec3::new(r[0] as f32, r[1] as f32, r[2] as f32);
    Ok(Alignment {
        transform: Similarity {
            rotation: [row(fit.rotation[0]), row(fit.rotation[1]), row(fit.rotation[2])],
            scale: fit.scale as f32,
            translation: row(fit.translation),
        },
        rms: fit.rms_error as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp25_similarity_recovers_a_known_transform() {
        let (c, s) = (0.6_f32.cos(), 0.6_f32.sin());
        let truth = Similarity {
            rotation: [Vec3::new(c, 0.0, s), Vec3::Y, Vec3::new(-s, 0.0, c)],
            scale: 1.07,
            translation: Vec3::new(0.02, -0.3, 0.1),
        };
        let source = [
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.12, 0.0),
            Vec3::new(0.0, 0.0, 0.09),
            Vec3::ONE * 0.05,
        ];
        let target: Vec<Vec3> = source.iter().map(|&p| truth.apply(p)).collect();
        let fit = fit_similarity(&source, &target, None, true).unwrap();
        assert!(fit.rms < 1e-6, "{}", fit.rms);
        assert!((fit.transform.scale - 1.07).abs() < 1e-5);
        for (&p, &q) in source.iter().zip(&target) {
            assert!(fit.transform.apply(p).abs_diff_eq(q, 1e-5));
            assert!(fit.transform.inverse().apply(q).abs_diff_eq(p, 1e-5));
        }
        let rigid = fit_similarity(&source, &target, None, false).unwrap();
        assert!((rigid.transform.scale - 1.0).abs() < 1e-9 && rigid.rms > 1e-4);
        assert!(fit_similarity(&source[..2], &target[..2], None, true).is_err());
        assert!(fit_similarity(&[Vec3::ZERO; 3], &target[..3], None, true).is_err());
    }
}
