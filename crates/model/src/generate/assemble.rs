//! Assembly of the template: the body surface and the parts become one [`Mesh`] with UV seams,
//! the region map is refined on the final faces `[MOD-03]`, and every table is put together.

use humanart_math::Vec3;

use super::Surface;
use super::anatomy::{Anatomy, sided};
use super::cage::Cage;
use super::parts::Part;
use super::regions::{C, R, region_table};
use super::uv::Layout;
use crate::error::{ModelError, Result};
use crate::mesh::{Mesh, MeshBuilder};
use crate::region::RegionMap;

/// The assembled mesh with the source of every vertex.
pub struct Assembled {
    /// The template mesh.
    pub mesh: Mesh,
    /// Per mesh vertex: (piece, vertex of the piece). Piece 0 is the body.
    pub source: Vec<(u32, u32)>,
    /// The region map.
    pub regions: RegionMap,
}

/// The refined region of each body face.
pub fn body_face_regions(mesh: &Cage, a: &Anatomy) -> Vec<R> {
    let to_head = |p: Vec3| (p - a.head_origin) * 1000.0;
    (0..mesh.face_count())
        .map(|f| {
            let tag = R::from_id(mesh.tags[f].region).unwrap_or(R::Body);
            let c = mesh.centroid(f);
            let h = to_head(c);
            let side = if c.x >= 0.0 { 1.0 } else { -1.0 };
            let lr = |l: R, r: R| R::sided(l, r, side);
            let e = (c - sided(a.eye_center, side)) * 1000.0;
            let (ex, ey) = (e.x * side, e.y);
            match tag {
                R::UpperLidL | R::UpperLidR | R::LowerLidL | R::LowerLidR => {
                    let medial = ((ex + 15.0).powi(2) + (ey + 0.5).powi(2)).sqrt();
                    let lateral = ((ex - 14.0).powi(2) + (ey - 2.0).powi(2)).sqrt();
                    if medial < 5.5 {
                        lr(R::InnerCanthusL, R::InnerCanthusR)
                    } else if lateral < 5.0 {
                        lr(R::OuterCanthusL, R::OuterCanthusR)
                    } else {
                        tag
                    }
                }
                R::Forehead | R::TempleL | R::TempleR | R::CheekL | R::CheekR
                    if (-18.0..24.0).contains(&ex) && (16.0..31.0).contains(&ey) && e.z > -6.0 =>
                {
                    lr(R::BrowL, R::BrowR)
                }
                R::CheekL | R::CheekR if h.y > -20.0 && h.x.abs() > 38.0 => lr(R::MalarL, R::MalarR),
                R::UpperLip if h.x.abs() < 6.5 && h.y > -38.5 => R::Philtrum,
                R::UpperLip | R::LowerLip if h.x.abs() > 21.0 => lr(R::LipCornerL, R::LipCornerR),
                R::Scalp if h.z > -10.0 && h.y < 102.0 => R::Hairline,
                _ => tag,
            }
        })
        .collect()
}

/// Per vertex of a piece, the region most of its faces have (ties: the lower region id).
fn vertex_regions(mesh: &Cage, face_regions: &[R]) -> Vec<R> {
    let mut counts: Vec<Vec<(R, u32)>> = vec![Vec::new(); mesh.pos.len()];
    for (f, &r) in face_regions.iter().enumerate() {
        for &v in mesh.corners_of(f) {
            let list = &mut counts[v as usize];
            match list.iter_mut().find(|e| e.0 == r) {
                Some(e) => e.1 += 1,
                None => list.push((r, 1)),
            }
        }
    }
    counts
        .iter()
        .map(|list| list.iter().max_by(|x, y| x.1.cmp(&y.1).then(y.0.cmp(&x.0))).map_or(R::Body, |e| e.0))
        .collect()
}

/// Builds the mesh and region map from the surface, the parts and the UV layout.
pub fn assemble(surface: &Surface, parts: &[Part], layout: &Layout) -> Result<Assembled> {
    let body = surface.mesh();
    let pieces: Vec<&Cage> = std::iter::once(body).chain(parts.iter().map(|p| &p.mesh)).collect();
    let body_faces = body_face_regions(body, &surface.anatomy);
    let mut piece_vertex_regions: Vec<Vec<R>> = vec![vertex_regions(body, &body_faces)];
    for p in parts {
        piece_vertex_regions.push(vec![p.region; p.mesh.pos.len()]);
    }
    let mut b = MeshBuilder::new();
    // The copies on the base of the male piece (chart `genital-base`, ADR-039) meet its outer
    // surface at a hard edge: the base faces into the body, so a shared normal would darken the
    // rim of the penis root and the scrotum.
    let mut hard = vec![false; layout.copies.len()];
    for (pi, mesh) in pieces.iter().enumerate() {
        for f in 0..mesh.face_count() {
            if mesh.tags[f].chart != C::GenitalBase.id() {
                continue;
            }
            for k in mesh.offsets[f] as usize..mesh.offsets[f + 1] as usize {
                if let Some(h) = layout.corner_copies[pi].get(k).and_then(|&c| hard.get_mut(c as usize)) {
                    *h = true;
                }
            }
        }
    }
    // One builder vertex per layout copy, in copy order: the first copy of a piece vertex is a
    // vertex, later ones are UV-seam duplicates of the first copy on the same side of a hard edge
    // (hard-edge duplicates of the other side's first copy when their side has none yet).
    let mut first: [Vec<Vec<u32>>; 2] =
        [0, 1].map(|_| pieces.iter().map(|m| vec![u32::MAX; m.pos.len()]).collect());
    let mut source = Vec::with_capacity(layout.copies.len());
    let mut vertex_regions = Vec::with_capacity(layout.copies.len());
    for (i, c) in layout.copies.iter().enumerate() {
        let (pi, v) = (c.piece as usize, c.vertex as usize);
        let p = pieces[pi].pos[v];
        let side = usize::from(hard[i]);
        let (own, other) = (first[side][pi][v], first[1 - side][pi][v]);
        let id = if own != u32::MAX {
            b.duplicate(own, c.uv, true)?
        } else if other != u32::MAX {
            b.duplicate(other, c.uv, false)?
        } else {
            b.vertex(p, c.uv)
        };
        if own == u32::MAX {
            first[side][pi][v] = id;
        }
        debug_assert_eq!(id as usize, source.len());
        source.push((c.piece, c.vertex));
        vertex_regions.push(piece_vertex_regions[pi][v].id());
    }
    let names: Vec<&str> = std::iter::once("body").chain(parts.iter().map(|p| p.name)).collect();
    // The genital faces (ADR-039) stay in the body part, which consumers treat as one closed skin
    // surface (the hair SDF, the validator); their regions (`R::is_genital`) tell them apart.
    for (pi, mesh) in pieces.iter().enumerate() {
        b.begin_part(names[pi], true)?;
        for f in 0..mesh.face_count() {
            let start = mesh.offsets[f] as usize;
            let corners: Vec<u32> =
                (start..mesh.offsets[f + 1] as usize).map(|k| layout.corner_copies[pi][k]).collect();
            let tile = corners
                .first()
                .and_then(|&c| layout.copies.get(c as usize))
                .map(|c| c.tile)
                .ok_or(ModelError::invalid("face", f, "no copy"))?;
            b.face(&corners, tile)?;
        }
    }
    let mesh = b.finish()?;
    let regions = RegionMap { regions: region_table(), vertex_regions, soft: None };
    Ok(Assembled { mesh, source, regions })
}
