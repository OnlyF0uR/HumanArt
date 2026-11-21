//! UV layout `[MOD-04]`: islands from the face charts, cuts to disks, conformal flattening, packing
//! into the tile table.
//!
//! 1. Islands are connected sets of faces with the same chart. A closed island (a part such as an
//!    eyeball or a tooth) is first split in two by the plane through its centroid across its
//!    shortest principal axis (eyeballs: front and back).
//! 2. Every island except the face (whose eye, mouth and nostril holes stay holes) is cut to a disk
//!    with a cut graph: a maximum spanning tree of the dual graph crosses the edges facing away from
//!    the island's seam direction first, so the remaining (cut) edges run where the surface faces
//!    the seam direction (under the arm, inside the leg, behind the neck); dangling cut edges are
//!    pruned.
//! 3. Each island is flattened by least-squares conformal maps (Lévy et al. 2002) on the same
//!    triangulation the validator uses; if any triangle flips, a Tutte embedding with the boundary on
//!    a circle (always flip-free for a disk) replaces it.
//! 4. Islands are scaled to their 3D area, rotated to their smallest bounding box and shelf-packed
//!    per tile with padding; one scale per tile keeps the texel density uniform in the tile. Left and
//!    right islands are separate (validation forbids mirrored or stacked UVs).

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use humanart_math::solve::{CsrMatrix, IncompleteCholesky, JacobiPreconditioner, PcgSettings, PcgWorkspace, pcg};
use humanart_math::{Vec2, Vec3};

use super::cage::Cage;
use super::regions::{C, CHARTS, T, TILES};
use crate::error::{ModelError, Result};

/// Padding between islands, in texels of the tile's resolution.
pub const PADDING_TEXELS: f32 = 12.0;

/// One vertex copy of the layout: a vertex of a piece with its UV and tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Copy {
    /// Piece index (0 = body, then the parts).
    pub piece: u32,
    /// Vertex of the piece.
    pub vertex: u32,
    /// Tile-local UV, origin top-left, V down.
    pub uv: Vec2,
    /// Tile index.
    pub tile: u16,
    /// Island index (global, in layout order).
    pub island: u32,
}

/// The UV layout of all pieces.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// All vertex copies.
    pub copies: Vec<Copy>,
    /// Per piece, the copy index of each face corner (aligned with the piece's `corners`).
    pub corner_copies: Vec<Vec<u32>>,
    /// Number of islands per tile.
    pub islands_per_tile: Vec<usize>,
}

/// The seam direction of a chart: cuts run where the surface faces this way.
fn seam_direction(chart: C) -> Vec3 {
    match chart {
        C::ArmL => Vec3::new(-0.6, -0.75, -0.25).normalize(),
        C::ArmR => Vec3::new(0.6, -0.75, -0.25).normalize(),
        C::LegL => Vec3::new(-0.8, 0.0, -0.6).normalize(),
        C::LegR => Vec3::new(0.8, 0.0, -0.6).normalize(),
        _ => -Vec3::Z,
    }
}

/// An island being flattened: its faces (piece-local face indices) in one piece.
struct Island {
    piece: usize,
    chart: C,
    faces: Vec<usize>,
}

/// Faces grouped by chart and connectivity.
fn islands_of(piece: usize, mesh: &Cage) -> Vec<Island> {
    let (_, index, edge_faces) = mesh.edges();
    let mut seen = vec![false; mesh.face_count()];
    let mut out = Vec::new();
    for start in 0..mesh.face_count() {
        if seen[start] {
            continue;
        }
        let chart = mesh.tags[start].chart;
        // Hands and feet: each digit (its own region) is its own island, so conformal flattening
        // does not crush the long digits.
        let by_region = CHARTS.get(chart as usize).is_some_and(|c| c.2 == T::HandsFeet);
        let region = mesh.tags[start].region;
        let same = |g: usize| mesh.tags[g].chart == chart && (!by_region || mesh.tags[g].region == region);
        let mut faces = vec![start];
        seen[start] = true;
        let mut i = 0;
        while i < faces.len() {
            let f = faces[i];
            i += 1;
            let c = mesh.corners_of(f);
            for k in 0..c.len() {
                let (a, b) = (c[k], c[(k + 1) % c.len()]);
                let e = index[&(a.min(b), a.max(b))] as usize;
                for &g in &edge_faces[e] {
                    let g = g as usize;
                    if g < seen.len() && !seen[g] && same(g) {
                        seen[g] = true;
                        faces.push(g);
                    }
                }
            }
        }
        let chart = CHARTS.get(chart as usize).map_or(C::Face, |c| c.0);
        out.push(Island { piece, chart, faces });
    }
    out
}

/// Edge -> faces within a face subset.
fn local_edges(mesh: &Cage, faces: &[usize]) -> HashMap<(u32, u32), Vec<usize>> {
    let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (li, &f) in faces.iter().enumerate() {
        let c = mesh.corners_of(f);
        for k in 0..c.len() {
            let (a, b) = (c[k], c[(k + 1) % c.len()]);
            map.entry((a.min(b), a.max(b))).or_default().push(li);
        }
    }
    map
}

/// Splits a closed island across its shortest axis (eyeballs: front and back); open islands are returned as they are.
fn split_closed(mesh: &Cage, island: Island) -> Vec<Island> {
    let edges = local_edges(mesh, &island.faces);
    if edges.values().any(|f| f.len() == 1) {
        return vec![island];
    }
    split_plane(mesh, &island, false)
}

/// Splits an island in two by the plane through its centroid across its longest (`longest`) or
/// shortest principal axis.
fn split_plane(mesh: &Cage, island: &Island, longest: bool) -> Vec<Island> {
    let centroids: Vec<Vec3> = island.faces.iter().map(|&f| mesh.centroid(f)).collect();
    let c = centroids.iter().fold(Vec3::ZERO, |s, p| s + *p) / centroids.len().max(1) as f32;
    // Longest axis by power iteration on the covariance.
    let mut cov = [[0.0_f32; 3]; 3];
    for p in &centroids {
        let d = (*p - c).to_array();
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x += d[i] * d[j];
            }
        }
    }
    let apply = |m: &[[f32; 3]; 3], v: Vec3| -> Vec3 {
        let a = v.to_array();
        Vec3::new(
            m[0][0] * a[0] + m[0][1] * a[1] + m[0][2] * a[2],
            m[1][0] * a[0] + m[1][1] * a[1] + m[1][2] * a[2],
            m[2][0] * a[0] + m[2][1] * a[1] + m[2][2] * a[2],
        )
    };
    // Largest eigenvalue by power iteration, then the smallest axis by iterating on
    // (largest * I - covariance).
    let mut v = Vec3::new(1.0, 0.7, 0.3).normalize();
    for _ in 0..60 {
        v = apply(&cov, v).normalize_or(v);
    }
    let largest = apply(&cov, v).dot(v);
    let mut shifted = cov;
    for (i, row) in shifted.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = if i == j { largest - *x } else { -*x };
        }
    }
    let mut axis = Vec3::new(0.3, 0.7, 1.0).normalize();
    for _ in 0..100 {
        axis = apply(&shifted, axis).normalize_or(axis);
    }
    if longest {
        axis = v;
    }
    // Eyeballs split front/back so the iris and cornea stay in one island.
    if !longest && matches!(island.chart, C::EyeballL | C::EyeballR) {
        axis = Vec3::Z;
    }
    let (mut front, mut back) = (Vec::new(), Vec::new());
    for (i, &f) in island.faces.iter().enumerate() {
        if (centroids[i] - c).dot(axis) >= 0.0 {
            front.push(f);
        } else {
            back.push(f);
        }
    }
    let mut out = Vec::new();
    for half in [front, back] {
        // Each half may fall apart into several components.
        out.extend(components(mesh, island.piece, island.chart, &half));
    }
    out
}

fn components(mesh: &Cage, piece: usize, chart: C, faces: &[usize]) -> Vec<Island> {
    let edges = local_edges(mesh, faces);
    let mut seen = vec![false; faces.len()];
    let mut out = Vec::new();
    for s in 0..faces.len() {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut stack = vec![s];
        let mut comp = Vec::new();
        while let Some(i) = stack.pop() {
            comp.push(faces[i]);
            let c = mesh.corners_of(faces[i]);
            for k in 0..c.len() {
                let (a, b) = (c[k], c[(k + 1) % c.len()]);
                for &j in &edges[&(a.min(b), a.max(b))] {
                    if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        comp.sort_unstable();
        out.push(Island { piece, chart, faces: comp });
    }
    out
}

/// A flattened island: local vertex copies (source vertex, uv) and per face the local corner
/// indices.
struct Flat {
    piece: usize,
    tile: T,
    verts: Vec<(u32, Vec2)>,
    faces: Vec<(usize, Vec<u32>)>,
}

/// Cuts an island to a disk (unless `keep_holes`) and returns local copies and faces.
fn cut(mesh: &Cage, island: &Island, keep_holes: bool) -> (Vec<u32>, Vec<Vec<u32>>) {
    let faces = &island.faces;
    let edges = local_edges(mesh, faces);
    let mut cut_edges: HashSet<(u32, u32)> = HashSet::new();
    if !keep_holes {
        // Maximum spanning tree of the dual graph (Prim), weights favour crossing visible edges.
        let seam = seam_direction(island.chart);
        let normals: Vec<Vec3> = faces
            .iter()
            .map(|&f| {
                let c = mesh.corners_of(f);
                let mut n = Vec3::ZERO;
                for k in 0..c.len() {
                    n += mesh.pos[c[k] as usize].cross(mesh.pos[c[(k + 1) % c.len()] as usize]);
                }
                n.normalize_or_zero()
            })
            .collect();
        let mut face_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); faces.len()];
        for (key, fs) in &edges {
            if fs.len() == 2 {
                face_edges[fs[0]].push(*key);
                face_edges[fs[1]].push(*key);
            }
        }
        for list in &mut face_edges {
            list.sort_unstable();
        }
        let mut in_tree = vec![false; faces.len()];
        let mut crossed: HashSet<(u32, u32)> = HashSet::new();
        // Heap of (weight, tie, edge, to-face); weights quantised for determinism.
        let mut heap: BinaryHeap<(i64, i64, (u32, u32), usize)> = BinaryHeap::new();
        let root = (0..faces.len())
            .min_by(|&a, &b| normals[a].dot(seam).total_cmp(&normals[b].dot(seam)).then(a.cmp(&b)))
            .unwrap_or(0);
        let push = |heap: &mut BinaryHeap<(i64, i64, (u32, u32), usize)>, from: usize, in_tree: &[bool]| {
            for &key in &face_edges[from] {
                let fs = &edges[&key];
                let to = if fs[0] == from { fs[1] } else { fs[0] };
                if !in_tree[to] {
                    let n = (normals[from] + normals[to]).normalize_or_zero();
                    let w = (-n.dot(seam) * 1.0e6) as i64;
                    heap.push((w, -((key.0 as i64) << 24 | key.1 as i64), key, to));
                }
            }
        };
        in_tree[root] = true;
        push(&mut heap, root, &in_tree);
        while let Some((_, _, key, to)) = heap.pop() {
            if in_tree[to] {
                continue;
            }
            in_tree[to] = true;
            crossed.insert(key);
            push(&mut heap, to, &in_tree);
        }
        for (key, fs) in &edges {
            if fs.len() == 2 && !crossed.contains(key) {
                cut_edges.insert(*key);
            }
        }
        // Prune dangling cut edges (an endpoint of degree 1 in cut + boundary edges).
        loop {
            let mut degree: HashMap<u32, u32> = HashMap::new();
            for (key, fs) in &edges {
                if fs.len() == 1 || cut_edges.contains(key) {
                    *degree.entry(key.0).or_default() += 1;
                    *degree.entry(key.1).or_default() += 1;
                }
            }
            let dangling: Vec<(u32, u32)> =
                cut_edges.iter().filter(|k| degree[&k.0] == 1 || degree[&k.1] == 1).copied().collect();
            if dangling.is_empty() {
                break;
            }
            for k in dangling {
                cut_edges.remove(&k);
            }
        }
    }
    // Vertex copies: faces around a vertex are joined across its non-cut interior edges.
    let mut incident: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (li, &f) in faces.iter().enumerate() {
        for &v in mesh.corners_of(f) {
            incident.entry(v).or_default().push(li);
        }
    }
    let mut copy_of: HashMap<(u32, usize), u32> = HashMap::new();
    let mut sources: Vec<u32> = Vec::new();
    for (&v, fl) in &incident {
        let mut parent: Vec<usize> = (0..fl.len()).collect();
        for (i, &fa) in fl.iter().enumerate() {
            for (j, &fb) in fl.iter().enumerate().skip(i + 1) {
                // Shared edges at v between faces fa and fb.
                let ca = mesh.corners_of(faces[fa]);
                let cb = mesh.corners_of(faces[fb]);
                let shares = ca.iter().any(|&w| {
                    w != v && cb.contains(&w) && {
                        let key = (v.min(w), v.max(w));
                        edges.get(&key).is_some_and(|f| f.len() == 2) && !cut_edges.contains(&key) && {
                            // w must be adjacent to v in both faces.
                            adjacent(ca, v, w) && adjacent(cb, v, w)
                        }
                    }
                });
                if shares {
                    let (x, y) = (find(&mut parent, i), find(&mut parent, j));
                    parent[x] = y;
                }
            }
        }
        let mut class_copy: HashMap<usize, u32> = HashMap::new();
        for (i, &fa) in fl.iter().enumerate() {
            let r = find(&mut parent, i);
            let id = *class_copy.entry(r).or_insert_with(|| {
                sources.push(v);
                (sources.len() - 1) as u32
            });
            copy_of.insert((v, fa), id);
        }
    }
    let local_faces: Vec<Vec<u32>> = faces
        .iter()
        .enumerate()
        .map(|(li, &f)| mesh.corners_of(f).iter().map(|&v| copy_of[&(v, li)]).collect())
        .collect();
    (sources, local_faces)
}

/// Union-find root with path halving.
fn find(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

fn adjacent(face: &[u32], a: u32, b: u32) -> bool {
    (0..face.len()).any(|k| {
        let (p, q) = (face[k], face[(k + 1) % face.len()]);
        (p == a && q == b) || (p == b && q == a)
    })
}

/// Splits faces into triangles like [`crate::mesh::Mesh::triangulate`] (quads on the shorter rest
/// diagonal, larger polygons as fans).
fn triangles(pos: &dyn Fn(u32) -> Vec3, faces: &[Vec<u32>]) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for f in faces {
        if let [a, b, c, d] = f[..] {
            if pos(b).distance_squared(pos(d)) < pos(a).distance_squared(pos(c)) {
                out.push([a, b, d]);
                out.push([b, c, d]);
            } else {
                out.push([a, b, c]);
                out.push([a, c, d]);
            }
        } else {
            for i in 1..f.len() - 1 {
                out.push([f[0], f[i], f[i + 1]]);
            }
        }
    }
    out
}

fn solve_spd(a: &CsrMatrix, b: &[f64], x: &mut [f64]) -> Result<()> {
    let settings = PcgSettings { tolerance: 1e-9, max_iterations: 20_000 };
    let mut ws = PcgWorkspace::new(a.rows());
    let done = if let Ok(ic) = IncompleteCholesky::new(a) {
        pcg(a, b, x, &ic, &settings, &mut ws)
    } else {
        let j =
            JacobiPreconditioner::new(a).map_err(|_| ModelError::invalid("uv solve", 0, "preconditioner failed"))?;
        pcg(a, b, x, &j, &settings, &mut ws)
    };
    done.map(|_| ()).map_err(|_| ModelError::invalid("uv solve", a.rows(), "conjugate gradients did not converge"))
}

/// Least-squares conformal map of a triangulated disk (or disk with holes).
fn lscm(p: &[Vec3], tris: &[[u32; 3]]) -> Result<Vec<Vec2>> {
    let n = p.len();
    // Pins: the two vertices farthest apart along the longest bounding-box axis.
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for q in p {
        lo = lo.min(*q);
        hi = hi.max(*q);
    }
    let ext = hi - lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        Vec3::X
    } else if ext.y >= ext.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let key = |i: usize| p[i].dot(axis);
    let p0 = (0..n).min_by(|&a, &b| key(a).total_cmp(&key(b))).unwrap_or(0);
    let p1 = (0..n).max_by(|&a, &b| key(a).total_cmp(&key(b))).unwrap_or(0);
    if p0 == p1 {
        return Err(ModelError::invalid("uv island", 0, "island has no extent"));
    }
    let d = f64::from(p[p0].distance(p[p1]));
    let pinned = |i: usize| -> Option<(f64, f64)> {
        if i == p0 {
            Some((0.0, 0.0))
        } else if i == p1 {
            Some((d, 0.0))
        } else {
            None
        }
    };
    // Unknown index of (vertex, coordinate).
    let mut idx = vec![usize::MAX; n];
    let mut m = 0;
    for (i, slot) in idx.iter_mut().enumerate() {
        if pinned(i).is_none() {
            *slot = m;
            m += 1;
        }
    }
    let mut trip: Vec<(usize, usize, f64)> = Vec::with_capacity(tris.len() * 72);
    let mut rhs = vec![0.0_f64; 2 * m];
    for t in tris {
        let [a, b, c] = t.map(|v| p[v as usize]);
        let e1 = (b - a).normalize_or_zero();
        let nrm = (b - a).cross(c - a);
        let area = 0.5 * f64::from(nrm.length());
        if area < 1e-14 {
            continue;
        }
        let e2 = nrm.normalize_or_zero().cross(e1);
        let xy = [
            (0.0_f64, 0.0_f64),
            (f64::from((b - a).length()), 0.0),
            (f64::from((c - a).dot(e1)), f64::from((c - a).dot(e2))),
        ];
        let s = 1.0 / (2.0 * area).sqrt();
        // Two rows: real and imaginary parts of sum_j W_j (u_j + i v_j).
        let mut rows: [[(usize, f64); 6]; 2] = [[(0, 0.0); 6]; 2];
        let mut fixed = [0.0_f64; 2];
        for j in 0..3 {
            let (k, l) = ((j + 1) % 3, (j + 2) % 3);
            let wa = (xy[k].0 - xy[l].0) * s;
            let wb = (xy[k].1 - xy[l].1) * s;
            let v = t[j] as usize;
            // Re: wa*u - wb*v ; Im: wb*u + wa*v
            let coef = [[wa, -wb], [wb, wa]];
            for r in 0..2 {
                if let Some((pu, pv)) = pinned(v) {
                    fixed[r] += coef[r][0] * pu + coef[r][1] * pv;
                    rows[r][2 * j] = (usize::MAX, 0.0);
                    rows[r][2 * j + 1] = (usize::MAX, 0.0);
                } else {
                    rows[r][2 * j] = (2 * idx[v], coef[r][0]);
                    rows[r][2 * j + 1] = (2 * idx[v] + 1, coef[r][1]);
                }
            }
        }
        for r in 0..2 {
            for &(i, ci) in &rows[r] {
                if i == usize::MAX {
                    continue;
                }
                rhs[i] -= ci * fixed[r];
                for &(j, cj) in &rows[r] {
                    if j != usize::MAX {
                        trip.push((i, j, ci * cj));
                    }
                }
            }
        }
    }
    let a = CsrMatrix::from_triplets(2 * m, 2 * m, &trip).map_err(|_| ModelError::invalid("uv island", 0, "matrix"))?;
    // Initial guess: projection onto the plane of the island's two largest extents.
    let mut x = vec![0.0_f64; 2 * m];
    for i in 0..n {
        if idx[i] != usize::MAX {
            let q = p[i] - p[p0];
            x[2 * idx[i]] = f64::from(q.dot(axis));
        }
    }
    solve_spd(&a, &rhs, &mut x)?;
    Ok((0..n)
        .map(|i| match pinned(i) {
            Some((u, v)) => Vec2::new(u as f32, v as f32),
            None => Vec2::new(x[2 * idx[i]] as f32, x[2 * idx[i] + 1] as f32),
        })
        .collect())
}

/// Tutte embedding with uniform weights: the (single) boundary loop on a circle by arc length.
fn tutte(p: &[Vec3], faces: &[Vec<u32>]) -> Result<Vec<Vec2>> {
    let n = p.len();
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        for k in 0..f.len() {
            *directed.entry((f[k], f[(k + 1) % f.len()])).or_default() += 1;
        }
    }
    let mut next: HashMap<u32, u32> = HashMap::new();
    for &(a, b) in directed.keys() {
        if !directed.contains_key(&(b, a)) {
            next.insert(a, b);
        }
    }
    let Some(&start) = next.keys().min() else {
        return Err(ModelError::invalid("uv island", 0, "island has no boundary"));
    };
    let mut loop_ = vec![start];
    let mut at = start;
    while let Some(&b) = next.get(&at) {
        if b == start || loop_.len() > n {
            break;
        }
        loop_.push(b);
        at = b;
    }
    let total: f32 =
        (0..loop_.len()).map(|i| p[loop_[i] as usize].distance(p[loop_[(i + 1) % loop_.len()] as usize])).sum();
    let radius = total / std::f32::consts::TAU;
    let mut fixed: HashMap<u32, Vec2> = HashMap::new();
    let mut s = 0.0;
    for i in 0..loop_.len() {
        let ang = std::f32::consts::TAU * s / total;
        fixed.insert(loop_[i], Vec2::new(radius * ang.cos(), radius * ang.sin()));
        s += p[loop_[i] as usize].distance(p[loop_[(i + 1) % loop_.len()] as usize]);
    }
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            if !nbrs[a as usize].contains(&b) {
                nbrs[a as usize].push(b);
            }
            if !nbrs[b as usize].contains(&a) {
                nbrs[b as usize].push(a);
            }
        }
    }
    let mut idx = vec![usize::MAX; n];
    let mut m = 0;
    for (i, slot) in idx.iter_mut().enumerate() {
        if !fixed.contains_key(&(i as u32)) {
            *slot = m;
            m += 1;
        }
    }
    let mut trip = Vec::new();
    let mut bu = vec![0.0_f64; m];
    let mut bv = vec![0.0_f64; m];
    for i in 0..n {
        if idx[i] == usize::MAX {
            continue;
        }
        trip.push((idx[i], idx[i], nbrs[i].len() as f64));
        for &j in &nbrs[i] {
            match fixed.get(&j) {
                Some(q) => {
                    bu[idx[i]] += f64::from(q.x);
                    bv[idx[i]] += f64::from(q.y);
                }
                None => trip.push((idx[i], idx[j as usize], -1.0)),
            }
        }
    }
    let mut out: Vec<Vec2> = (0..n).map(|i| fixed.get(&(i as u32)).copied().unwrap_or(Vec2::ZERO)).collect();
    if m > 0 {
        let a = CsrMatrix::from_triplets(m, m, &trip).map_err(|_| ModelError::invalid("uv island", 0, "matrix"))?;
        let mut xu = vec![0.0; m];
        let mut xv = vec![0.0; m];
        solve_spd(&a, &bu, &mut xu)?;
        solve_spd(&a, &bv, &mut xv)?;
        for i in 0..n {
            if idx[i] != usize::MAX {
                out[i] = Vec2::new(xu[idx[i]] as f32, xv[idx[i]] as f32);
            }
        }
    }
    Ok(out)
}

/// The boundary loops of a polygon set (each loop in walking order).
fn boundary_loops(faces: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        for k in 0..f.len() {
            *directed.entry((f[k], f[(k + 1) % f.len()])).or_default() += 1;
        }
    }
    let mut next: BTreeMap<u32, u32> = BTreeMap::new();
    for &(a, b) in directed.keys() {
        if !directed.contains_key(&(b, a)) {
            next.insert(a, b);
        }
    }
    let mut loops = Vec::new();
    while let Some((&start, _)) = next.iter().next() {
        let mut loop_ = vec![start];
        let mut at = start;
        while let Some(b) = next.remove(&at) {
            if b == start {
                break;
            }
            loop_.push(b);
            at = b;
        }
        loops.push(loop_);
    }
    loops
}

/// Virtual triangles closing every hole (all boundary loops but the longest), by ear clipping in
/// the hole's best-fit plane. The boundary runs the opposite way round a hole, so the triangles
/// are listed against the loop's walking order to keep the island's orientation.
fn hole_fill(p: &[Vec3], faces: &[Vec<u32>]) -> Vec<[u32; 3]> {
    let loops = boundary_loops(faces);
    let len = |l: &Vec<u32>| -> f32 {
        (0..l.len()).map(|i| p[l[i] as usize].distance(p[l[(i + 1) % l.len()] as usize])).sum()
    };
    let Some(outer) = (0..loops.len()).max_by(|&a, &b| len(&loops[a]).total_cmp(&len(&loops[b]))) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (li, l) in loops.iter().enumerate() {
        if li == outer || l.len() < 3 {
            continue;
        }
        // Reverse so the hole polygon winds like a face of the island.
        let poly: Vec<u32> = l.iter().rev().copied().collect();
        let c = poly.iter().fold(Vec3::ZERO, |s, &v| s + p[v as usize]) / poly.len() as f32;
        let mut n = Vec3::ZERO;
        for i in 0..poly.len() {
            n += (p[poly[i] as usize] - c).cross(p[poly[(i + 1) % poly.len()] as usize] - c);
        }
        let n = n.normalize_or(Vec3::Z);
        // Zipper fill: split the hole at its two farthest-apart vertices into two chains and
        // stitch them by arc length (slit-shaped holes get fat, symmetric triangles).
        let m = poly.len();
        let (mut i0, mut i1, mut best) = (0, 0, -1.0_f32);
        for i in 0..m {
            for j in i + 1..m {
                let d = p[poly[i] as usize].distance_squared(p[poly[j] as usize]);
                if d > best {
                    (i0, i1, best) = (i, j, d);
                }
            }
        }
        let chain_a: Vec<u32> = (i0..=i1).map(|k| poly[k]).collect();
        let chain_b: Vec<u32> = (0..=(m - i1 + i0)).map(|k| poly[(i0 + m - k) % m]).collect();
        let arc = |c: &[u32]| -> Vec<f32> {
            let mut acc = vec![0.0_f32];
            for w in c.windows(2) {
                let last = acc[acc.len() - 1];
                acc.push(last + p[w[0] as usize].distance(p[w[1] as usize]));
            }
            let total = acc[acc.len() - 1].max(1e-12);
            acc.iter().map(|x| x / total).collect()
        };
        let (ta, tb) = (arc(&chain_a), arc(&chain_b));
        let (mut i, mut j) = (0, 0);
        while i + 1 < chain_a.len() || j + 1 < chain_b.len() {
            let advance_a = j + 1 >= chain_b.len() || (i + 1 < chain_a.len() && ta[i + 1] <= tb[j + 1]);
            let tri = if advance_a {
                i += 1;
                [chain_a[i - 1], chain_a[i], chain_b[j]]
            } else {
                j += 1;
                [chain_a[i], chain_b[j], chain_b[j - 1]]
            };
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                continue;
            }
            let [x, y, z] = tri.map(|v| p[v as usize]);
            let tri = if (y - x).cross(z - x).dot(n) < 0.0 { [tri[0], tri[2], tri[1]] } else { tri };
            out.push(tri);
        }
    }
    out
}

/// Whether any two triangles of a flattened island overlap (not counting shared edges), with a
/// uniform grid over the triangles' bounding boxes.
fn self_overlaps(uv: &[Vec2], tris: &[[u32; 3]]) -> bool {
    if tris.len() < 2 {
        return false;
    }
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for p in uv {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let ext = (hi - lo).max(Vec2::splat(1e-9));
    let side = ((tris.len() as f32).sqrt().ceil() as usize).clamp(1, 512);
    let cell = |p: Vec2| -> (usize, usize) {
        let q = (p - lo) / ext * side as f32;
        ((q.x.max(0.0) as usize).min(side - 1), (q.y.max(0.0) as usize).min(side - 1))
    };
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); side * side];
    for (i, t) in tris.iter().enumerate() {
        let c = t.map(|v| uv[v as usize]);
        let (a, b) = (cell(c[0].min(c[1]).min(c[2])), cell(c[0].max(c[1]).max(c[2])));
        for y in a.1..=b.1 {
            for x in a.0..=b.0 {
                grid[y * side + x].push(i);
            }
        }
    }
    let tol = 1e-6 * ext.x.max(ext.y);
    for members in &grid {
        for (k, &i) in members.iter().enumerate() {
            for &j in &members[k + 1..] {
                let (a, b) = (tris[i].map(|v| uv[v as usize]), tris[j].map(|v| uv[v as usize]));
                if overlap(&a, &b, tol) {
                    return true;
                }
            }
        }
    }
    false
}

/// Separating-axis test of two triangles' interiors with a tolerance.
fn overlap(a: &[Vec2; 3], b: &[Vec2; 3], tol: f32) -> bool {
    for tri in [a, b] {
        for i in 0..3 {
            let e = tri[(i + 1) % 3] - tri[i];
            let Some(n) = Vec2::new(-e.y, e.x).try_normalize() else { continue };
            let proj = |t: &[Vec2; 3]| {
                let d = [n.dot(t[0]), n.dot(t[1]), n.dot(t[2])];
                (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
            };
            let ((a0, a1), (b0, b1)) = (proj(a), proj(b));
            if a1 <= b0 + tol || b1 <= a0 + tol {
                return false;
            }
        }
    }
    true
}

fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    0.5 * (b - a).perp_dot(c - a)
}

/// Flattens one island.
fn flatten(mesh: &Cage, island: &Island, may_split: bool) -> Result<Option<Flat>> {
    let keep_holes = island.chart == C::Face;
    let (sources, faces) = cut(mesh, island, keep_holes);
    let p: Vec<Vec3> = sources.iter().map(|&v| mesh.pos[v as usize]).collect();
    let tris = triangles(&|v| p[v as usize], &faces);
    let ok = |uv: &[Vec2]| {
        tris.iter().all(|t| signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) > 0.0)
            && !self_overlaps(uv, &tris)
    };
    // Holes (all boundary loops but the longest) are filled with virtual triangles for the solve,
    // so their sharp corners (lip corners, canthi) are interior to the conformal map.
    let mut solve_tris = tris.clone();
    if keep_holes {
        solve_tris.extend(hole_fill(&p, &faces));
    }
    let mut uv = lscm(&p, &solve_tris).unwrap_or_default();
    if uv.len() != p.len() || !ok(&uv) {
        // Islands that fold under the conformal map are split and tried again; only small ones
        // fall back to a Tutte embedding.
        if may_split && island.faces.len() > 16 {
            return Ok(None);
        }
        uv = tutte(&p, &faces)?;
        if !ok(&uv) {
            return Err(ModelError::invalid(
                "uv island",
                island.faces.len(),
                "island could not be flattened without flips",
            ));
        }
    }
    // Scale to the 3D area.
    let (mut a3, mut a2) = (0.0_f64, 0.0_f64);
    for t in &tris {
        let [x, y, z] = t.map(|v| p[v as usize]);
        a3 += 0.5 * f64::from((y - x).cross(z - x).length());
        a2 += f64::from(signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]));
    }
    let s = if a2 > 0.0 { (a3 / a2).sqrt() as f32 } else { 1.0 };
    let verts = sources.iter().zip(&uv).map(|(&v, &q)| (v, q * s)).collect();
    let faces = island.faces.iter().copied().zip(faces).collect();
    Ok(Some(Flat { piece: island.piece, tile: island.chart.tile(), verts, faces }))
}

/// Rotates an island to its smallest bounding box; returns (rotated points, width, height).
fn orient(points: &[Vec2]) -> (Vec<Vec2>, f32, f32) {
    let mut best: Option<(f32, f32)> = None;
    for step in 0..24 {
        let ang = (step as f32 * 7.5).to_radians();
        let (s, c) = ang.sin_cos();
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in points {
            let q = Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y);
            lo = lo.min(q);
            hi = hi.max(q);
        }
        let area = (hi.x - lo.x) * (hi.y - lo.y);
        if best.is_none_or(|b| area < b.1 - 1e-12) {
            best = Some((ang, area));
        }
    }
    let ang = best.map_or(0.0, |b| b.0);
    let (s, c) = ang.sin_cos();
    let rotated: Vec<Vec2> = points.iter().map(|p| Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y)).collect();
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for q in &rotated {
        lo = lo.min(*q);
        hi = hi.max(*q);
    }
    let (w, h) = (hi.x - lo.x, hi.y - lo.y);
    // Tall islands stand upright: rotate by 90 degrees (keeps orientation) when wider than tall.
    let out: Vec<Vec2> = if w > h {
        rotated.iter().map(|q| Vec2::new(-(q.y - lo.y), q.x - lo.x) + Vec2::new(h, 0.0)).collect()
    } else {
        rotated.iter().map(|q| *q - lo).collect()
    };
    let (w, h) = if w > h { (h, w) } else { (w, h) };
    (out, w, h)
}

/// Shelf packing of boxes `(w, h)` with padding `pad`, row width `limit`: offsets and total size.
fn shelf(boxes: &[(f32, f32)], pad: f32, limit: f32) -> Shelves {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[b].1.total_cmp(&boxes[a].1).then(a.cmp(&b)));
    let mut out = vec![Vec2::ZERO; boxes.len()];
    let (mut x, mut y, mut row_h, mut width) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for i in order {
        let (w, h) = boxes[i];
        if x > 0.0 && x + w > limit {
            y += row_h + pad;
            x = 0.0;
            row_h = 0.0;
        }
        out[i] = Vec2::new(x, y);
        x += w + pad;
        row_h = row_h.max(h);
        width = width.max(x - pad);
    }
    (out, width, y + row_h)
}

/// A shelf packing: the boxes' offsets, the width and the height.
type Shelves = (Vec<Vec2>, f32, f32);

/// Shelf widths tried (plus one) when packing the face tile.
const PACK_WIDTHS: usize = 28;

/// Lays out UVs for the pieces (body first, then the parts).
pub fn layout(pieces: &[&Cage]) -> Result<Layout> {
    let mut flats: Vec<Flat> = Vec::new();
    for (pi, mesh) in pieces.iter().enumerate() {
        for island in islands_of(pi, mesh) {
            let mut queue: Vec<(Island, usize)> = split_closed(mesh, island).into_iter().map(|i| (i, 0)).collect();
            while let Some((part, depth)) = queue.pop() {
                match flatten(mesh, &part, depth < 4)? {
                    Some(flat) => flats.push(flat),
                    None => queue.extend(split_plane(mesh, &part, true).into_iter().map(|i| (i, depth + 1))),
                }
            }
        }
    }
    let mut copies: Vec<Copy> = Vec::new();
    let mut corner_copies: Vec<Vec<u32>> = pieces.iter().map(|m| vec![u32::MAX; m.corners.len()]).collect();
    let mut islands_per_tile = vec![0; TILES.len()];
    for (ti, &(tile, _, resolution, _)) in TILES.iter().enumerate() {
        let members: Vec<usize> = (0..flats.len()).filter(|&i| flats[i].tile == tile).collect();
        islands_per_tile[ti] = members.len();
        if members.is_empty() {
            continue;
        }
        let oriented: Vec<(Vec<Vec2>, f32, f32)> =
            members.iter().map(|&i| orient(&flats[i].verts.iter().map(|v| v.1).collect::<Vec<_>>())).collect();
        let boxes: Vec<(f32, f32)> = oriented.iter().map(|o| (o.1, o.2)).collect();
        let area: f32 = boxes.iter().map(|b| b.0 * b.1).sum();
        let widest = boxes.iter().map(|b| b.0).fold(0.0, f32::max);
        // Face tile: shelf widths from 0.6 to 2.0 times the side of the islands' total area, the
        // one that gives the largest scale wins (the single width left more than half of the tile
        // empty and the nostrils under 1000 texels at 1024², `ac_wp15_2`). The other tiles keep
        // the single width 1.08 for now: repacked, the navel's one-texel faces land far from the
        // tile origin, where `Mesh::face_uv_area` (shoelace on absolute f32 UVs) loses them to
        // cancellation and the validator calls them collapsed. Three passes per width: the
        // padding depends on the final scale.
        let margin = PADDING_TEXELS / resolution as f32;
        let mut best: Option<(f32, Shelves)> = None;
        let widths = if tile == T::Face { PACK_WIDTHS } else { 0 };
        for step in 0..=widths {
            let factor = if widths == 0 { 1.08 } else { 0.6 + 1.4 * step as f32 / widths as f32 };
            let limit = (area.sqrt() * factor).max(widest);
            let mut scale = 0.8 / area.sqrt();
            let mut placed = (Vec::new(), 0.0, 0.0);
            for _ in 0..3 {
                let pad = PADDING_TEXELS / (resolution as f32 * scale);
                placed = shelf(&boxes, pad, limit);
                let extent = placed.1.max(placed.2);
                scale = (1.0 - 2.0 * margin) / extent;
            }
            if best.as_ref().is_none_or(|b| scale > b.0) {
                best = Some((scale, placed));
            }
        }
        let Some((scale, placed)) = best else { continue };
        for (k, &fi) in members.iter().enumerate() {
            let flat = &flats[fi];
            let base = copies.len() as u32;
            let offset = placed.0[k];
            for (j, &(v, _)) in flat.verts.iter().enumerate() {
                let q = (oriented[k].0[j] + offset) * scale + Vec2::splat(margin);
                // Math frame (V up) to tile frame (V down).
                copies.push(Copy {
                    piece: flat.piece as u32,
                    vertex: v,
                    uv: Vec2::new(q.x, 1.0 - q.y),
                    tile: ti as u16,
                    island: fi as u32,
                });
            }
            for (f, local) in &flat.faces {
                let mesh = pieces[flat.piece];
                let start = mesh.offsets[*f] as usize;
                for (c, &l) in local.iter().enumerate() {
                    corner_copies[flat.piece][start + c] = base + l;
                }
            }
        }
    }
    if corner_copies.iter().any(|c| c.contains(&u32::MAX)) {
        return Err(ModelError::invalid("uv layout", 0, "a face corner has no UV"));
    }
    Ok(Layout { copies, corner_copies, islands_per_tile })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::cage::Tag;

    /// A closed cylinder (radius 0.1, height 0.4) along Y with grid caps, all in one chart.
    fn cylinder(chart: C) -> Cage {
        let mut c = Cage::new();
        let n = 16;
        let rings: Vec<Vec<u32>> = (0..5)
            .map(|k| {
                let pts: Vec<Vec3> = (0..n)
                    .map(|i| {
                        let a = std::f32::consts::TAU * i as f32 / n as f32;
                        Vec3::new(0.1 * a.cos(), 0.1 * k as f32, -0.1 * a.sin())
                    })
                    .collect();
                c.add_all(&pts)
            })
            .collect();
        let tag = Tag::new(0, chart.id());
        for w in rings.windows(2) {
            c.bridge(&w[0], &w[1], tag);
        }
        let rev: Vec<u32> = rings[0].iter().take(1).chain(rings[0].iter().skip(1).rev()).copied().collect();
        c.cap(&rev, 4, 4, 0, tag, &|_, _, p| p).unwrap();
        c.cap(&rings[4], 4, 4, 0, tag, &|_, _, p| p).unwrap();
        c
    }

    #[test]
    fn mod_04_lscm_flattens_a_plane_without_distortion() {
        // A tilted planar grid: the conformal map is an isometry up to scale.
        let (u, v) = (Vec3::new(0.6, 0.0, 0.8), Vec3::new(0.0, 1.0, 0.0));
        let p: Vec<Vec3> = (0..36).map(|i| u * (i % 6) as f32 * 0.01 + v * (i / 6) as f32 * 0.013).collect();
        let faces: Vec<Vec<u32>> =
            (0..25).map(|q| (q % 5 + 6 * (q / 5)) as u32).map(|a| vec![a, a + 1, a + 7, a + 6]).collect();
        let tris = triangles(&|i| p[i as usize], &faces);
        let uv = lscm(&p, &tris).unwrap();
        let s = p[0].distance(p[35]) / uv[0].distance(uv[35]);
        for (a, b) in [(0_usize, 5_usize), (0, 30), (7, 28), (12, 17)] {
            let (d3, d2) = (p[a].distance(p[b]), uv[a].distance(uv[b]) * s);
            assert!((d3 - d2).abs() < 1e-4 * d3.max(1e-3), "{a}-{b}: {d3} vs {d2}");
        }
    }

    #[test]
    fn mod_04_closed_cylinder_is_cut_split_and_packed_without_flips_or_overlaps() {
        let c = cylinder(C::ArmL);
        let layout = layout(&[&c]).unwrap();
        assert!(layout.copies.iter().all(|k| (0.0..=1.0).contains(&k.uv.x) && (0.0..=1.0).contains(&k.uv.y)));
        let uv: Vec<Vec2> = layout.copies.iter().map(|k| k.uv).collect();
        let faces: Vec<Vec<u32>> = (0..c.face_count())
            .map(|f| (c.offsets[f] as usize..c.offsets[f + 1] as usize).map(|k| layout.corner_copies[0][k]).collect())
            .collect();
        let tris = triangles(&|i| uv[i as usize].extend(0.0), &faces);
        // Tile frame has V down: faces that are counter-clockwise from outside are clockwise there.
        assert!(tris.iter().all(|t| signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) < 0.0));
        assert!(!self_overlaps(&uv, &tris));
        assert!(layout.islands_per_tile[T::Arms as usize] >= 1);
    }

    #[test]
    fn mod_04_shelf_packing_keeps_padding_between_boxes() {
        let mut rng = humanart_core::rng::Rng::new(7);
        let boxes: Vec<(f32, f32)> = (0..60).map(|_| (rng.range_f32(0.01, 0.21), rng.range_f32(0.01, 0.21))).collect();
        let pad = 0.01;
        let (at, w, h) = shelf(&boxes, pad, 1.0);
        for i in 0..boxes.len() {
            assert!(at[i].x + boxes[i].0 <= w + 1e-6 && at[i].y + boxes[i].1 <= h + 1e-6);
            for j in i + 1..boxes.len() {
                let apart = at[i].x + boxes[i].0 + pad <= at[j].x + 1e-6
                    || at[j].x + boxes[j].0 + pad <= at[i].x + 1e-6
                    || at[i].y + boxes[i].1 + pad <= at[j].y + 1e-6
                    || at[j].y + boxes[j].1 + pad <= at[i].y + 1e-6;
                assert!(apart, "boxes {i} and {j} overlap");
            }
        }
    }
}
