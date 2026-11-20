//! The template as a fit surface `[WP-25]`: which faces are fitted, how their UV-seam copies are
//! welded into nodes, how strongly each node is pulled by the scan, and where the template's
//! landmarks sit on the nodes.
//!
//! **Selection.** The faces of one mesh part (the skin, `body`) whose corners all lie in the
//! selected regions (each with its subtree): for a head fit `head` and `neck`. Eyeballs, teeth and
//! the other separate parts are never fitted: a scan shows them fused into the skin, if at all.
//!
//! **Weld.** Corners with the same position key (the copies of one vertex across a UV seam) become
//! one node, so the fit cannot tear a seam open. Triangles that collapse under the weld are dropped.
//!
//! **Node weights.** Each node takes the weight of the deepest region listed in
//! [`RegionWeights`] that contains its vertex's region (the default for none): 0 where the scan
//! does not show the template's surface (the mouth interior, the eye sockets), less than 1 where
//! it shows something on top of it (a swim cap over the scalp).
//!
//! **Landmarks.** A template landmark is a barycentric point on three vertices; it maps to the
//! same point on their nodes, and is skipped when a vertex with a non-zero weight is not fitted.

use humanart_math::Vec3;
use humanart_model::landmark::Landmark;
use humanart_model::mesh::Mesh;
use humanart_model::region::RegionMap;

use super::register::{FitLandmark, FitMesh};
use super::{CaptureError, Result};
use crate::bake::bvh::Bvh;
use crate::bake::sampling::{cosine_hemisphere, point, rotation, texel_hash, to_world};

/// The marker of a template vertex that is not fitted.
pub const NOT_FITTED: u32 = u32::MAX;

/// Node weights by region: the deepest listed region containing a node's region decides.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionWeights {
    /// `(region name, weight in 0..=1)`.
    pub weights: Vec<(String, f32)>,
    /// The weight of a node in no listed region.
    pub default: f32,
}

impl RegionWeights {
    /// The weights of a head fit onto a scan whose subject wears a swim cap and has fused,
    /// closed eyes (HSRD-100 style): the face, ears and neck pull fully; the scalp and hairline
    /// a little (the cap and the hair under it sit a few millimetres above the skull); the eye
    /// sockets and the mouth interior not at all (the scan has no surface there).
    #[must_use]
    pub fn head_scan() -> Self {
        let w = |name: &str, weight: f32| (name.to_owned(), weight);
        Self {
            weights: vec![
                w("scalp", 0.15),
                w("hairline", 0.5),
                w("eye-socket-L", 0.0),
                w("eye-socket-R", 0.0),
                w("mouth-interior", 0.0),
                w("mouth-helper", 0.0),
                w("teeth-upper", 0.0),
                w("teeth-lower", 0.0),
                w("gums-upper", 0.0),
                w("gums-lower", 0.0),
                w("tongue", 0.0),
            ],
            default: 1.0,
        }
    }

    /// The weight of region `region` of `regions`.
    #[must_use]
    pub fn weight(&self, regions: &RegionMap, region: u16) -> f32 {
        let mut at = Some(region);
        // Parents have lower indices: the walk ends after at most `regions.len()` steps.
        for _ in 0..=regions.regions.len() {
            let Some(r) = at else { break };
            let Some(info) = regions.regions.get(r as usize) else { break };
            if let Some((_, w)) = self.weights.iter().find(|(name, _)| *name == info.name) {
                return *w;
            }
            at = info.parent;
        }
        self.default
    }
}

/// The selected template faces as a welded fit surface.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateFit {
    /// The welded nodes at the template's rest positions and their triangles.
    pub mesh: FitMesh,
    /// The data weight of each node (`0..=1`).
    pub node_weights: Vec<f32>,
    /// For each node, one template vertex at its position (the first copy).
    pub node_vertex: Vec<u32>,
    /// For each template vertex, its node or [`NOT_FITTED`].
    pub vertex_node: Vec<u32>,
    /// The template vertices of each fitted triangle's corners (unwelded: these carry the UVs),
    /// parallel to `mesh.triangles`.
    pub corners: Vec<u32>,
    /// The template face of each fitted triangle.
    pub triangle_faces: Vec<u32>,
}

impl TemplateFit {
    /// The faces of mesh part `part` whose corners all lie in one of `select` (region names, each
    /// with its subtree), welded by position key, with node weights from `weights`.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] for an unknown part or region, a region map that does not
    /// match the mesh, or a selection without faces; [`CaptureError::Limit`] beyond
    /// [`super::register::MAX_FIT_NODES`].
    pub fn new(mesh: &Mesh, regions: &RegionMap, part: &str, select: &[&str], weights: &RegionWeights) -> Result<Self> {
        let vertex_count = mesh.vertex_count();
        if regions.vertex_regions.len() != vertex_count || mesh.position_keys().len() != vertex_count {
            return Err(CaptureError::invalid("the region map does not match the template mesh"));
        }
        let part = mesh
            .parts()
            .iter()
            .find(|p| p.name == part)
            .ok_or(CaptureError::invalid("the template has no such mesh part"))?;
        let faces = part.faces();
        let mut roots = Vec::with_capacity(select.len());
        for name in select {
            roots.push(regions.find(name).ok_or(CaptureError::invalid("the template has no such region"))?);
        }
        let selected = |v: u32| {
            regions.vertex_regions.get(v as usize).is_some_and(|&r| roots.iter().any(|&a| regions.is_within(r, a)))
        };
        let tri = mesh.triangulate();
        let positions = mesh.positions();
        let keys = mesh.position_keys();
        let mut key_node = vec![NOT_FITTED; vertex_count];
        let mut vertex_node = vec![NOT_FITTED; vertex_count];
        let mut fit = FitMesh::default();
        let (mut node_vertex, mut node_weights, mut corners, mut triangle_faces) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (t, &face) in tri.triangle_faces.iter().enumerate() {
            let Some(c) = tri.triangle(t) else { continue };
            if !faces.contains(&(face as usize)) || !c.iter().all(|&v| selected(v)) {
                continue;
            }
            let mut nodes = [0_u32; 3];
            for (k, &v) in c.iter().enumerate() {
                let key = keys.get(v as usize).copied().ok_or(CaptureError::invalid("a template vertex is missing"))?;
                let slot =
                    key_node.get_mut(key as usize).ok_or(CaptureError::invalid("a position key is out of range"))?;
                if *slot == NOT_FITTED {
                    if fit.positions.len() >= super::register::MAX_FIT_NODES {
                        return Err(CaptureError::limit(
                            "fit node count",
                            super::register::MAX_FIT_NODES,
                            fit.positions.len() + 1,
                        ));
                    }
                    *slot = fit.positions.len() as u32;
                    let p = positions
                        .get(v as usize)
                        .copied()
                        .ok_or(CaptureError::invalid("a template vertex is missing"))?;
                    fit.positions.push(p);
                    node_vertex.push(v);
                    let region = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
                    node_weights.push(weights.weight(regions, region).clamp(0.0, 1.0));
                }
                nodes[k] = *slot;
                if let Some(vn) = vertex_node.get_mut(v as usize) {
                    *vn = *slot;
                }
            }
            if nodes[0] == nodes[1] || nodes[1] == nodes[2] || nodes[0] == nodes[2] {
                continue;
            }
            fit.triangles.extend_from_slice(&nodes);
            corners.extend_from_slice(&c);
            triangle_faces.push(face);
        }
        if fit.triangles.is_empty() {
            return Err(CaptureError::invalid("the selection holds no faces"));
        }
        // Copies of a fitted vertex outside the selected faces (the same position key) follow it.
        for (v, slot) in vertex_node.iter_mut().enumerate() {
            if *slot == NOT_FITTED
                && let Some(&n) = keys.get(v).and_then(|&k| key_node.get(k as usize))
            {
                *slot = n;
            }
        }
        Ok(Self { mesh: fit, node_weights, node_vertex, vertex_node, corners, triangle_faces })
    }

    /// Scales every node weight by how visible the node is from outside: the cosine-weighted
    /// fraction of `rays` hemisphere rays around its normal that leave the whole template `mesh`
    /// (all parts: eyeballs and teeth occlude too) within `max_distance`, mapped linearly from
    /// `lo` (weight × 0) to `hi` (weight × 1). A scan only captures what a camera sees: the inner
    /// lips, the nostrils, the lid margins facing the eyeball and the deep ear folds of the
    /// template have no scan surface, and pulling them onto the outer surface folds them. Returns
    /// the visibility of every node.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] for bad parameters; [`CaptureError::Bake`] when the ray
    /// hierarchy cannot be built.
    pub fn weight_by_visibility(
        &mut self,
        mesh: &Mesh,
        rays: u32,
        max_distance: f32,
        lo: f32,
        hi: f32,
    ) -> Result<Vec<f32>> {
        if !((1..=4096).contains(&rays) && max_distance > 0.0 && max_distance.is_finite() && hi > lo) {
            return Err(CaptureError::invalid("visibility needs 1..=4096 rays, a positive distance and lo < hi"));
        }
        let tri = mesh.triangulate();
        let bvh = Bvh::build(mesh.positions(), &tri.indices).map_err(CaptureError::Bake)?;
        let normals = self.mesh.normals(&self.mesh.positions);
        let mut out = Vec::with_capacity(self.mesh.positions.len());
        for (n, (&p, &nrm)) in self.mesh.positions.iter().zip(&normals).enumerate() {
            let r = rotation(texel_hash(n as u32, 1, 0x7a11));
            // Start 0.1 mm off the surface (template edges are a millimetre or more).
            let origin = p + nrm * 1e-4;
            let open = (0..rays)
                .filter(|&i| {
                    !bvh.occluded(origin, to_world(cosine_hemisphere(point(i, rays, r)), nrm), 0.0, max_distance)
                })
                .count();
            let v = open as f32 / rays as f32;
            if let Some(w) = self.node_weights.get_mut(n) {
                *w *= ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            }
            out.push(v);
        }
        Ok(out)
    }

    /// The node positions after mapping every rest position through `f` (e.g. the alignment into
    /// the scan's frame).
    #[must_use]
    pub fn mapped_positions(&self, f: impl Fn(Vec3) -> Vec3) -> Vec<Vec3> {
        self.mesh.positions.iter().map(|&p| f(p)).collect()
    }

    /// The fit landmark of template landmark `landmark` pulled to `target`, or `None` when it does
    /// not lie on the fitted nodes.
    #[must_use]
    pub fn landmark(&self, landmark: &Landmark, target: Vec3) -> Option<FitLandmark> {
        let mapped = landmark.vertices.map(|v| self.vertex_node.get(v as usize).copied().unwrap_or(NOT_FITTED));
        let any = mapped.iter().copied().find(|&n| n != NOT_FITTED)?;
        let mut nodes = [any; 3];
        for ((node, &m), &w) in nodes.iter_mut().zip(&mapped).zip(&landmark.weights) {
            if m != NOT_FITTED {
                *node = m;
            } else if w.abs() > 1e-6 {
                return None;
            }
        }
        Some(FitLandmark { nodes, weights: landmark.weights, target })
    }

    /// The template landmark position on the rest nodes (`None` when it is not fitted).
    #[must_use]
    pub fn landmark_rest(&self, landmark: &Landmark) -> Option<Vec3> {
        self.landmark(landmark, Vec3::ZERO).map(|l| l.position(&self.mesh.positions))
    }

    /// The template positions with the fitted node positions `nodes` written into every fitted
    /// vertex (all copies of a node's position key); the other vertices keep `rest`.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] when `nodes` or `rest` do not match this fit.
    pub fn apply(&self, rest: &[Vec3], nodes: &[Vec3]) -> Result<Vec<Vec3>> {
        if nodes.len() != self.mesh.positions.len() || rest.len() != self.vertex_node.len() {
            return Err(CaptureError::invalid("the positions do not match the fit"));
        }
        Ok(rest
            .iter()
            .zip(&self.vertex_node)
            .map(|(&p, &n)| if n == NOT_FITTED { p } else { nodes.get(n as usize).copied().unwrap_or(p) })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_math::Vec2;
    use humanart_model::landmark::LandmarkGroup;
    use humanart_model::mesh::MeshBuilder;
    use humanart_model::names::Side;
    use humanart_model::region::Region;

    /// A 4 × 2 quad strip (part `skin`, a UV seam between columns 1 and 2) plus a one-triangle
    /// part `eye`; regions: body > head > {face, scalp}, and body > leg.
    fn fixture() -> (Mesh, RegionMap) {
        let mut b = MeshBuilder::new();
        b.begin_part("skin", false).unwrap();
        let mut grid = [[0_u32; 5]; 3];
        for (j, row) in grid.iter_mut().enumerate() {
            for (i, v) in row.iter_mut().enumerate() {
                *v = b.vertex(Vec3::new(i as f32, j as f32, 0.0), Vec2::new(i as f32 / 8.0, j as f32 / 2.0));
            }
        }
        // Column 2 again on the far side of a seam.
        let seam: Vec<u32> =
            (0..3).map(|j| b.duplicate(grid[j][2], Vec2::new(0.9, j as f32 / 2.0), true).unwrap()).collect();
        for j in 0..2 {
            for i in 0..4 {
                let c = |jj: usize, ii: usize| if ii == 2 && i >= 2 { seam[jj] } else { grid[jj][ii] };
                b.face(&[c(j, i), c(j, i + 1), c(j + 1, i + 1), c(j + 1, i)], 0).unwrap();
            }
        }
        b.begin_part("eye", true).unwrap();
        let e =
            [b.vertex(Vec3::Z, Vec2::ZERO), b.vertex(Vec3::X + Vec3::Z, Vec2::X), b.vertex(Vec3::Y + Vec3::Z, Vec2::Y)];
        b.face(&e, 1).unwrap();
        let mesh = b.finish().unwrap();
        let region = |name: &str, parent| Region { name: name.into(), parent, side: Side::Center };
        let regions = vec![
            region("body", None),
            region("head", Some(0)),
            region("face", Some(1)),
            region("scalp", Some(1)),
            region("leg", Some(0)),
        ];
        // Column 4 is leg, column 3 scalp, rows 0..=2 of columns 0..=2 face; the eye is face.
        let vertex_regions = mesh
            .positions()
            .iter()
            .enumerate()
            .map(|(v, p)| {
                if v >= 18 {
                    2
                } else if p.x > 3.5 {
                    4
                } else if p.x > 2.5 {
                    3
                } else {
                    2
                }
            })
            .collect();
        (mesh, RegionMap { regions, vertex_regions, soft: None })
    }

    #[test]
    fn wp25_head_selection_welds_seams_and_weights_regions() {
        let (mesh, regions) = fixture();
        let weights = RegionWeights { weights: vec![("scalp".into(), 0.25)], default: 1.0 };
        let fit = TemplateFit::new(&mesh, &regions, "skin", &["head"], &weights).unwrap();
        // Columns 0..=3 of the strip: 3 quads per row (the leg column and the eye are out), 12
        // triangles; nodes: 4 columns x 3 rows, the seam copies welded.
        assert_eq!(fit.mesh.triangles.len(), 3 * 12);
        assert_eq!(fit.mesh.positions.len(), 12);
        assert_eq!(fit.corners.len(), fit.mesh.triangles.len());
        for (j, row) in [0_u32, 5, 10].iter().enumerate() {
            let seam_copy = 15 + j as u32;
            assert_eq!(fit.vertex_node[(row + 2) as usize], fit.vertex_node[seam_copy as usize]);
            assert_ne!(fit.vertex_node[(row + 2) as usize], NOT_FITTED);
            assert_eq!(fit.vertex_node[(row + 4) as usize], NOT_FITTED, "leg column");
        }
        assert!(fit.vertex_node[18..].iter().all(|&n| n == NOT_FITTED), "eye part");
        for (n, &v) in fit.node_vertex.iter().enumerate() {
            let expect = if mesh.positions()[v as usize].x > 2.5 { 0.25 } else { 1.0 };
            assert_eq!(fit.node_weights[n], expect);
        }
        // Applying moved nodes moves every copy and nothing else.
        let moved: Vec<Vec3> = fit.mesh.positions.iter().map(|p| *p + Vec3::Z).collect();
        let out = fit.apply(mesh.positions(), &moved).unwrap();
        assert_eq!(out[15], out[2]);
        assert_eq!(out[2].z, 1.0);
        assert_eq!(out[4], mesh.positions()[4]);
        assert!(fit.apply(mesh.positions(), &moved[1..]).is_err());
    }

    #[test]
    fn wp25_template_landmarks_map_to_nodes_or_are_skipped() {
        let (mesh, regions) = fixture();
        let fit = TemplateFit::new(&mesh, &regions, "skin", &["face"], &RegionWeights::head_scan()).unwrap();
        let on =
            Landmark { name: "a".into(), group: LandmarkGroup::Face, vertices: [0, 1, 17], weights: [0.5, 0.5, 0.0] };
        let rest = fit.landmark_rest(&on).unwrap();
        assert!((rest - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-6);
        let target = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(fit.landmark(&on, target).unwrap().target, target);
        // Vertex 4 (leg) carries weight: skipped.
        let off = Landmark { vertices: [0, 4, 1], weights: [0.5, 0.5, 0.0], ..on.clone() };
        assert!(fit.landmark(&off, target).is_none());
        // Visibility: the open strip sees its whole hemisphere (+Z side); the eye triangle 1 m in
        // front of it hides part of it for nodes under it.
        let mut fit2 = fit.clone();
        let vis = fit2.weight_by_visibility(&mesh, 64, 0.5, 0.2, 0.6).unwrap();
        let far = fit2.vertex_node[0] as usize;
        assert_eq!(vis[far], 1.0);
        assert!(vis.iter().all(|v| (0.0..=1.0).contains(v)));
        let mut fit3 = fit.clone();
        let vis_near = fit3.weight_by_visibility(&mesh, 256, 2.0, 0.2, 0.6).unwrap();
        let under = fit3.vertex_node[0] as usize;
        assert!(vis_near[under] < 1.0, "{}", vis_near[under]);
        assert!(fit.clone().weight_by_visibility(&mesh, 0, 0.5, 0.2, 0.6).is_err());
        // Bad requests fail cleanly.
        assert!(TemplateFit::new(&mesh, &regions, "nope", &["face"], &RegionWeights::head_scan()).is_err());
        assert!(TemplateFit::new(&mesh, &regions, "skin", &["nope"], &RegionWeights::head_scan()).is_err());
        assert!(TemplateFit::new(&mesh, &regions, "eye", &["leg"], &RegionWeights::head_scan()).is_err());
    }
}
