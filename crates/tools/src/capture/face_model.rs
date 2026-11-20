//! Building the face shape model `[WP-27]` (todo 1.2) from several template-to-scan fits: the
//! mean face as an offset field and the principal components of the subjects around it, written
//! as [`humanart_model::scanfit::face_model::FaceModel`].
//!
//! **Alignment.** Each subject's fitted nodes are already in the template's frame and at the
//! template's head size (the fit's landmark similarity, scale included, is undone). They are
//! aligned once more to the template's rest surface by a weighted rigid transform over all nodes
//! ([`super::align::fit_similarity`], weights = the caller's per-node alignment weights × the
//! fit's data weights), so the placement uses the dense surface instead of a few hand-picked
//! landmarks. [`BuildSettings::align_scale`] lets this step rescale too; it is off by default
//! because a dense fit of a deep real face to a flatter template shrinks it (measured 0.91..0.996
//! on the first 8 HSRD-100 subjects), which moves the eyes and teeth.
//!
//! **Mean.** Per node, the offset `aligned − rest` averaged over the subjects, each weighted by its
//! fit's data weight at that node floored at [`BuildSettings::min_weight`]. The default floor 1 is
//! the plain mean: where a scan showed nothing (hair over the ears, a collar, the eye sockets) the
//! fit already put a smooth extrapolation, and weighting by the per-node visibility weights mixes
//! different subsets of subjects on neighbouring nodes, which folds thin parts such as the ears.
//! Optionally made left-right symmetric (the template is symmetric; asymmetry is left to the
//! components).
//!
//! **Components.** PCA ([`humanart_math::solve::pca::Pca`]) of the subjects' offsets around the
//! mean. `N` subjects give at most `N − 1` components.
//!
//! **Blend.** Mean and components are multiplied by the caller's per-node blend mask before they
//! are stored, so the generator adds them as they are.

use humanart_math::Vec3;
use humanart_math::solve::dense::DenseMatrix;
use humanart_math::solve::pca::Pca;
use humanart_model::scanfit::face_model::{Component, FaceModel};
use humanart_model::scanfit::fitted::FittedSurface;

use super::align::fit_similarity;
use super::nearest::TriangleTree;
use super::{CaptureError, Result};

/// The most subjects one build takes.
pub const MAX_SUBJECTS: usize = 256;

/// How the model is built.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildSettings {
    /// The most components kept.
    pub components: usize,
    /// The floor of a subject's node weight in the mean (1: the plain mean).
    pub min_weight: f32,
    /// Whether the dense alignment may rescale a subject (module documentation).
    pub align_scale: bool,
    /// Whether the mean is made left-right symmetric (mirror `x ↦ −x`).
    pub symmetrise: bool,
    /// How far a mirrored node may lie from the rest surface, metres.
    pub mirror_distance: f32,
    /// Most untangling rounds of the stored mean ([`untangle`]; 0: none).
    pub untangle_rounds: u32,
}

impl Default for BuildSettings {
    /// Up to 9 components, the plain mean, rigid dense alignment, symmetric mean within 5 mm, up
    /// to 200 untangling rounds.
    fn default() -> Self {
        Self {
            components: 9,
            min_weight: 1.0,
            align_scale: false,
            symmetrise: true,
            mirror_distance: 0.005,
            untangle_rounds: 200,
        }
    }
}

/// What the build measured for one subject.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectReport {
    /// The fit's source scan.
    pub source: String,
    /// The scale of the dense alignment (fitted to rest; 1 unless [`BuildSettings::align_scale`]).
    pub scale: f32,
    /// Weighted RMS distance of the aligned subject from the rest surface, metres.
    pub rms_from_template: f32,
    /// Weighted RMS distance of the aligned subject from the mean, metres.
    pub rms_from_mean: f32,
}

/// What the build measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildReport {
    /// Per subject.
    pub subjects: Vec<SubjectReport>,
    /// Per kept component: its share of the total variance.
    pub variance_shares: Vec<f32>,
    /// Per kept component: the RMS node offset of `+1` SD after the blend, metres.
    pub component_rms: Vec<f32>,
    /// RMS node offset of the stored mean (after blend), metres.
    pub mean_rms: f32,
    /// Largest node offset of the stored mean, metres.
    pub mean_max: f32,
    /// Largest left-right difference of the mean before symmetrising, metres.
    pub asymmetry_max: f32,
    /// Flipped triangles of `rest + mean` before untangling.
    pub mean_flips_before: usize,
    /// Flipped triangles of `rest + mean` as stored.
    pub mean_flips: usize,
    /// Untangling rounds run.
    pub untangle_rounds: u32,
}

/// The triangles whose orientation flips from `rest` to `rest + offsets`.
#[must_use]
pub fn flipped(rest: &[Vec3], triangles: &[u32], offsets: &[Vec3]) -> Vec<u32> {
    let at = |x: &[Vec3], i: u32| x.get(i as usize).copied().unwrap_or(Vec3::ZERO);
    let moved = |i: u32| at(rest, i) + at(offsets, i);
    let mut out = Vec::new();
    for (t, tri) in triangles.as_chunks::<3>().0.iter().enumerate() {
        let [a, b, c] = *tri;
        let before = (at(rest, b) - at(rest, a)).cross(at(rest, c) - at(rest, a));
        let after = (moved(b) - moved(a)).cross(moved(c) - moved(a));
        if before.length_squared() > 0.0 && before.dot(after) <= 0.0 {
            out.push(t as u32);
        }
    }
    out
}

/// Removes orientation flips of `rest + offsets` locally: each round, the offsets of the nodes of
/// the flipped triangles and their neighbours move halfway to their neighbours' mean (umbrella
/// smoothing of the offset field only where it folds), until nothing flips or `rounds` ran.
/// Returns the flips before, after, and the rounds run.
pub fn untangle(rest: &[Vec3], triangles: &[u32], offsets: &mut [Vec3], rounds: u32) -> (usize, usize, u32) {
    let n = rest.len().min(offsets.len());
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in triangles.as_chunks::<3>().0 {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if let Some(l) = adj.get_mut(a as usize) {
                l.push(b);
            }
            if let Some(l) = adj.get_mut(b as usize) {
                l.push(a);
            }
        }
    }
    let before = flipped(rest, triangles, offsets).len();
    let mut flips = before;
    let mut marked = vec![false; n];
    let mut next = offsets.to_vec();
    let mut ran = 0;
    while flips > 0 && ran < rounds {
        ran += 1;
        marked.fill(false);
        for t in flipped(rest, triangles, offsets) {
            for k in 0..3 {
                if let Some(&v) = triangles.get(3 * t as usize + k) {
                    for &j in std::iter::once(&v).chain(adj.get(v as usize).map_or(&[][..], Vec::as_slice)) {
                        if let Some(m) = marked.get_mut(j as usize) {
                            *m = true;
                        }
                    }
                }
            }
        }
        next.copy_from_slice(offsets);
        for (i, nb) in adj.iter().enumerate() {
            if !marked[i] || nb.is_empty() {
                continue;
            }
            let mean = nb
                .iter()
                .map(|&j| offsets.get(j as usize).copied().unwrap_or(Vec3::ZERO))
                .fold(Vec3::ZERO, |a, b| a + b)
                / nb.len() as f32;
            next[i] = (offsets[i] + mean) * 0.5;
        }
        offsets.copy_from_slice(&next);
        flips = flipped(rest, triangles, offsets).len();
    }
    (before, flips, ran)
}

fn check_same(first: &FittedSurface, other: &FittedSurface) -> Result<()> {
    if other.nodes.len() != first.nodes.len()
        || other.triangles != first.triangles
        || other.template_name != first.template_name
        || other.template_version != first.template_version
    {
        return Err(CaptureError::invalid("the fits do not share one template surface"));
    }
    if !other
        .nodes
        .iter()
        .zip(&first.nodes)
        .all(|(a, b)| a.template_vertex == b.template_vertex && a.rest.abs_diff_eq(b.rest, 1.0e-6))
    {
        return Err(CaptureError::invalid("the fits' rest positions differ (different template builds)"));
    }
    Ok(())
}

fn rms(values: impl Iterator<Item = (f64, f64)>) -> f32 {
    let (mut sum, mut weight) = (0.0, 0.0);
    for (d2, w) in values {
        sum += d2 * w;
        weight += w;
    }
    if weight > 0.0 { (sum / weight).sqrt() as f32 } else { 0.0 }
}

const fn mirror(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}

/// Builds the model (module documentation). `blend` and `align_weights` hold one value in `0..=1`
/// per fit node; `description` names the sources and their attribution.
///
/// # Errors
/// [`CaptureError::Invalid`] for fewer than 2 fits, fits of different template surfaces or
/// malformed masks; [`CaptureError::Limit`] beyond [`MAX_SUBJECTS`]; the errors of the
/// alignment, the PCA and the closest-point tree.
pub fn build(
    fits: &[FittedSurface],
    blend: &[f32],
    align_weights: &[f32],
    description: String,
    settings: &BuildSettings,
) -> Result<(FaceModel, BuildReport)> {
    let Some(first) = fits.first() else {
        return Err(CaptureError::invalid("a face model needs at least 2 fits"));
    };
    if fits.len() < 2 {
        return Err(CaptureError::invalid("a face model needs at least 2 fits"));
    }
    if fits.len() > MAX_SUBJECTS {
        return Err(CaptureError::limit("face model subject count", MAX_SUBJECTS, fits.len()));
    }
    let n = first.nodes.len();
    let in_unit = |m: &[f32]| m.len() == n && m.iter().all(|w| (0.0..=1.0).contains(w));
    if !in_unit(blend) || !in_unit(align_weights) {
        return Err(CaptureError::invalid("the blend and alignment masks need one value in 0..=1 per node"));
    }
    for f in fits {
        f.validate()?;
        check_same(first, f)?;
    }
    let rest: Vec<Vec3> = first.nodes.iter().map(|v| v.rest).collect();

    // Dense alignment of each subject onto the rest surface.
    let mut aligned: Vec<Vec<Vec3>> = Vec::with_capacity(fits.len());
    let mut report = BuildReport::default();
    for f in fits {
        let fitted: Vec<Vec3> = f.nodes.iter().map(|v| v.fitted).collect();
        let w: Vec<f64> = f.nodes.iter().zip(align_weights).map(|(v, &a)| f64::from(v.weight * a)).collect();
        let t = fit_similarity(&fitted, &rest, Some(&w), settings.align_scale)?.transform;
        let x: Vec<Vec3> = fitted.iter().map(|&p| t.apply(p)).collect();
        report.subjects.push(SubjectReport {
            source: f.source.clone(),
            scale: t.scale,
            rms_from_template: rms(x
                .iter()
                .zip(&rest)
                .zip(&w)
                .map(|((p, r), &w)| (f64::from(p.distance_squared(*r)), w))),
            rms_from_mean: 0.0,
        });
        aligned.push(x);
    }

    // Weighted mean offset.
    let weight = |k: usize, i: usize| {
        fits.get(k).and_then(|f| f.nodes.get(i)).map_or(0.0, |v| v.weight.max(settings.min_weight))
    };
    let mut mean = vec![Vec3::ZERO; n];
    for (i, m) in mean.iter_mut().enumerate() {
        let (mut sum, mut total) = (Vec3::ZERO, 0.0_f32);
        for (k, x) in aligned.iter().enumerate() {
            let w = weight(k, i);
            sum += (x[i] - rest[i]) * w;
            total += w;
        }
        *m = if total > 0.0 { sum / total } else { Vec3::ZERO };
    }
    for (k, x) in aligned.iter().enumerate() {
        if let Some(s) = report.subjects.get_mut(k) {
            s.rms_from_mean = rms((0..n).map(|i| {
                let d = x[i] - rest[i] - mean[i];
                (f64::from(d.length_squared()), f64::from(weight(k, i) * align_weights[i]))
            }));
        }
    }

    // Components: PCA of the subjects around the mean.
    let rows: Vec<Vec<f64>> = aligned
        .iter()
        .map(|x| {
            let mut row = Vec::with_capacity(3 * n);
            for i in 0..n {
                let d = (x[i] - rest[i] - mean[i]) * blend[i];
                row.extend([f64::from(d.x), f64::from(d.y), f64::from(d.z)]);
            }
            row
        })
        .collect();
    // Identical subjects leave nothing to decompose (and the PCA refuses zero variance).
    let varies = rows.iter().skip(1).any(|r| r.iter().zip(&rows[0]).any(|(a, b)| (a - b).abs() > 1.0e-9));
    let pca = if varies { Some(Pca::fit(&DenseMatrix::from_rows(&rows)?)?) } else { None };
    let kept = pca.as_ref().map_or(0, |p| settings.components.min(p.component_count()));
    let mut components = Vec::with_capacity(kept);
    for j in 0..kept {
        let Some(pca) = pca.as_ref() else { break };
        let (Some(c), Some(&var)) = (pca.component(j), pca.variances().get(j)) else { break };
        let sd = var.max(0.0).sqrt();
        let offsets: Vec<Vec3> = c
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| Vec3::new((v[0] * sd) as f32, (v[1] * sd) as f32, (v[2] * sd) as f32))
            .collect();
        report.variance_shares.push(pca.explained_variance_ratio(j) as f32);
        report.component_rms.push(rms(offsets.iter().map(|o| (f64::from(o.length_squared()), 1.0))));
        components.push(Component { variance: pca.explained_variance_ratio(j).clamp(0.0, 1.0) as f32, offsets });
    }

    // Symmetric mean.
    if settings.symmetrise {
        let tree = TriangleTree::build(&rest, &first.triangles)?;
        let mut sym = mean.clone();
        for (i, s) in sym.iter_mut().enumerate() {
            let Some(hit) = tree.nearest(mirror(rest[i]), settings.mirror_distance, |_| true) else { continue };
            let t = hit.triangle as usize;
            let Some(tri) = first.triangles.get(3 * t..3 * t + 3) else { continue };
            let other = tri
                .iter()
                .zip(hit.bary)
                .map(|(&j, b)| mean.get(j as usize).copied().unwrap_or(Vec3::ZERO) * b)
                .fold(Vec3::ZERO, |a, b| a + b);
            report.asymmetry_max = report.asymmetry_max.max(mean[i].distance(mirror(other)));
            *s = (mean[i] + mirror(other)) * 0.5;
        }
        mean = sym;
    }
    let mut mean: Vec<Vec3> = mean.iter().zip(blend).map(|(&m, &b)| m * b).collect();
    let (before, after, ran) = untangle(&rest, &first.triangles, &mut mean, settings.untangle_rounds);
    (report.mean_flips_before, report.mean_flips, report.untangle_rounds) = (before, after, ran);
    report.mean_rms = rms(mean.iter().map(|m| (f64::from(m.length_squared()), 1.0)));
    report.mean_max = mean.iter().map(|m| m.length()).fold(0.0, f32::max);

    let model = FaceModel {
        description,
        template_name: first.template_name.clone(),
        template_version: first.template_version,
        subjects: fits.len() as u32,
        template_vertices: first.nodes.iter().map(|v| v.template_vertex).collect(),
        rest,
        mean,
        components,
        triangles: first.triangles.clone(),
    };
    model.validate()?;
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use humanart_model::scanfit::Similarity;
    use humanart_model::scanfit::fitted::{FittedNode, FittedSummary};

    use super::*;

    /// A symmetric curved grid "face" (x in -2..=2 cm), its fit displaced by `shape`
    /// and placed by `place`.
    fn fit(shape: impl Fn(Vec3) -> Vec3, place: Similarity, weight: impl Fn(Vec3) -> f32) -> FittedSurface {
        let mut nodes = Vec::new();
        for j in 0..9 {
            for i in 0..9 {
                // A gently curved "face" (a plane would leave the rigid fit undetermined).
                let (x, y) = ((i as f32 - 4.0) * 0.005, j as f32 * 0.005);
                let rest = Vec3::new(x, y, -20.0 * x * x - 10.0 * (y - 0.02) * (y - 0.02));
                nodes.push(FittedNode {
                    template_vertex: (j * 9 + i) as u32,
                    rest,
                    fitted: place.apply(rest + shape(rest)),
                    error: 0.0,
                    weight: weight(rest),
                    matched: true,
                });
            }
        }
        let mut triangles = Vec::new();
        for j in 0..8_u32 {
            for i in 0..8_u32 {
                let a = j * 9 + i;
                triangles.extend([a, a + 1, a + 10, a, a + 10, a + 9]);
            }
        }
        FittedSurface {
            source: "test".into(),
            attribution: String::new(),
            template_name: "t".into(),
            template_version: 1,
            template_vertices: 81,
            alignment: Similarity::IDENTITY,
            summary: FittedSummary::default(),
            nodes,
            triangles,
        }
    }

    #[test]
    fn wp27_mean_ignores_placement_and_unreliable_nodes() {
        // The same shape three times: once as is, once rotated, scaled and moved (the dense
        // alignment removes that), once with a corner the scan did not show (data weight 0,
        // displaced 1 cm) that must hardly pull the mean.
        let bump = |p: Vec3| Vec3::Z * 0.002 * (1.0 - (p.x / 0.02).powi(2)) + Vec3::Y * 0.0005 * (p.y / 0.04);
        let corner = |p: Vec3| p.x > 0.012 && p.y > 0.03;
        let (c, s) = (0.3_f32.cos(), 0.3_f32.sin());
        let moved = Similarity {
            rotation: [Vec3::new(c, -s, 0.0), Vec3::new(s, c, 0.0), Vec3::Z],
            scale: 1.2,
            translation: Vec3::new(0.1, 0.0, 0.3),
        };
        let all = |_: Vec3| 1.0;
        let fits = vec![
            fit(bump, Similarity::IDENTITY, all),
            fit(bump, moved, all),
            fit(
                |p| if corner(p) { Vec3::Z * 0.01 } else { bump(p) },
                Similarity::IDENTITY,
                |p| if corner(p) { 0.0 } else { 1.0 },
            ),
        ];
        let n = fits[0].nodes.len();
        let ones = vec![1.0; n];
        let settings =
            BuildSettings { symmetrise: false, min_weight: 0.02, align_scale: true, ..BuildSettings::default() };
        let (reference, _) = build(&[fits[0].clone(), fits[0].clone()], &ones, &ones, "ref".into(), &settings).unwrap();
        let (model, report) = build(&fits, &ones, &ones, "test".into(), &settings).unwrap();
        assert_eq!(model.subjects, 3);
        // The moved subject needs 1/1.2 more scale than the first (which absorbs a little of the
        // shape as scale itself).
        assert!((report.subjects[1].scale / report.subjects[0].scale - 1.0 / 1.2).abs() < 1e-3, "{report:?}");
        assert!(report.subjects[0].rms_from_template > 0.0005, "{report:?}");
        for (i, (m, r)) in model.mean.iter().zip(&reference.mean).enumerate() {
            assert!(m.abs_diff_eq(*r, 0.00015), "node {i}: {m:?} vs {r:?}");
        }
        assert!(model.components.len() <= 2, "3 subjects give at most 2 components");
        let back = FaceModel::parse(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(back.rest, model.rest);

        // Untangling: a node pushed through its neighbours flips triangles; local smoothing of the
        // offsets removes the flips and leaves the rest of the field alone.
        let mut offsets = model.mean.clone();
        let centre = 4 * 9 + 4;
        offsets[centre] += Vec3::new(0.012, 0.0, 0.0);
        assert!(!flipped(&model.rest, &model.triangles, &offsets).is_empty());
        let far = offsets[0];
        let (before, after, rounds) = untangle(&model.rest, &model.triangles, &mut offsets, 100);
        assert!(before > 0 && after == 0 && rounds > 0, "{before} {after} {rounds}");
        assert_eq!(offsets[0], far, "nodes away from the fold keep their offsets");

        // Malformed input.
        assert!(build(&fits[..1], &ones, &ones, String::new(), &settings).is_err());
        assert!(build(&fits, &ones[1..], &ones, String::new(), &settings).is_err());
        let mut other = fits[1].clone();
        other.nodes[3].rest.x += 0.001;
        assert!(build(&[fits[0].clone(), other], &ones, &ones, String::new(), &settings).is_err());
    }

    #[test]
    fn wp27_one_varying_shape_is_one_component_and_symmetrising_mirrors() {
        let wave = |a: f32| move |p: Vec3| Vec3::Z * a * (p.y * 150.0).sin() * (1.0 - (p.x / 0.02).powi(2));
        let all = |_: Vec3| 1.0;
        let fits: Vec<FittedSurface> =
            [0.001, -0.001, 0.0005, 0.0].iter().map(|&a| fit(wave(a), Similarity::IDENTITY, all)).collect();
        let n = fits[0].nodes.len();
        let ones = vec![1.0; n];
        let (model, report) = build(&fits, &ones, &ones, "test".into(), &BuildSettings::default()).unwrap();
        assert!(report.variance_shares[0] > 0.95, "{report:?}");
        assert!(report.component_rms[0] > 0.0002, "{report:?}");
        assert!(model.components.len() <= 3);

        // A bump on one side only: symmetrising puts half of it on each side.
        let side =
            |p: Vec3| Vec3::Z * 0.003 * (-((p.x - 0.015) / 0.004).powi(2) - ((p.y - 0.02) / 0.006).powi(2)).exp();
        let fits = vec![fit(side, Similarity::IDENTITY, all), fit(side, Similarity::IDENTITY, all)];
        let (model, report) = build(&fits, &ones, &ones, "test".into(), &BuildSettings::default()).unwrap();
        assert!(report.asymmetry_max > 0.002, "{report:?}");
        // Grid nodes (i, j) = (7, 4) and (1, 4): x = ±15 mm, y = 20 mm.
        let (right, left) = (model.mean[4 * 9 + 7], model.mean[4 * 9 + 1]);
        assert!(right.z > 0.0008 && (right.z - left.z).abs() < 0.0001, "{right:?} {left:?}");
        assert!((right.x + left.x).abs() < 0.0001, "{right:?} {left:?}");
    }
}
