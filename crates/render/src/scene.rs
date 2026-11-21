//! What the renderer draws: meshes, plain PBR materials and objects, the procedural test scenes,
//! and their GPU form with the previous frame's positions `[RND-04]`.
//!
//! Vertex data is uploaded as separate streams (positions, previous positions, normals, uvs,
//! indices: structure of arrays `[PERF-R2]`) into one buffer per stream shared by all meshes, so a
//! frame binds the scene once and draws each object with an index range and a base vertex. The
//! shaders pull vertices from the streams (`shaders/render/scene.wgsl`).
//!
//! Motion vectors: before new positions arrive, the frame encoder copies the current positions of
//! that mesh into the previous-position stream (and once more on the frame after, so a mesh that
//! stopped moving has zero motion). Objects keep their previous transform likewise.
//!
//! **Deformed meshes** `[EVAL-05]`: [`GpuScene::attach_deformer`] hands a mesh's position,
//! previous-position and normal streams over to a [`GpuDeformer`]: the passes then pull that mesh's
//! vertices straight from the deformer's output buffers (no copy), through a group-1 bind group of
//! its own that the renderer builds per deformed mesh ([`ExternalStreams`]). Its draws use base
//! vertex 0 (the deformer's buffers hold only that mesh), its texture coordinates are copied once
//! into a buffer of their own, and [`GpuScene::begin_frame`] no longer keeps its previous
//! positions: the deformer does (`GpuDeformer::encode` copies them before it deforms).

use humanart_gpu::pod::slice_as_bytes;
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, Tag};
use humanart_math::{Aabb, Mat4, Sphere, Vec2, Vec3};

use crate::focus::CasterSlabs;

use crate::deform::GpuDeformer;
use crate::error::RenderError;
use crate::sss::{SKIN_CLASS, SkinShading};

/// Most vertices of one scene (all meshes together).
pub const MAX_VERTICES: usize = 1 << 24;
/// Most indices of one scene.
pub const MAX_INDICES: usize = 3 << 24;
/// Most objects of one scene.
pub const MAX_OBJECTS: usize = 4096;

/// A triangle mesh: per-vertex streams and a triangle list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    /// Positions, metres.
    pub positions: Vec<Vec3>,
    /// Unit normals.
    pub normals: Vec<Vec3>,
    /// Texture coordinates (origin top-left, V down).
    pub uvs: Vec<Vec2>,
    /// Three indices per triangle, counter-clockwise seen from the front.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Checks the streams: equal lengths, whole triangles, indices in range, finite values, the
    /// documented limits.
    pub fn validate(&self) -> Result<(), RenderError> {
        let n = self.positions.len();
        if n > MAX_VERTICES {
            return Err(RenderError::TooLarge { what: "vertices", value: n as u64, limit: MAX_VERTICES as u64 });
        }
        if self.indices.len() > MAX_INDICES {
            return Err(RenderError::TooLarge {
                what: "indices",
                value: self.indices.len() as u64,
                limit: MAX_INDICES as u64,
            });
        }
        if self.normals.len() != n || self.uvs.len() != n {
            return Err(RenderError::InvalidMesh { what: "positions, normals and uvs differ in length" });
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(RenderError::InvalidMesh { what: "index count is not a multiple of 3" });
        }
        if self.indices.iter().any(|&i| i as usize >= n) {
            return Err(RenderError::InvalidMesh { what: "an index is past the last vertex" });
        }
        if !self.positions.iter().chain(&self.normals).all(|p| p.is_finite()) || !self.uvs.iter().all(|u| u.is_finite())
        {
            return Err(RenderError::InvalidMesh { what: "a vertex attribute is not finite" });
        }
        Ok(())
    }

    /// The bounding box of the positions.
    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(&self.positions)
    }

    /// An ellipsoid with radii `radii` around `center`: `segments` around Y, `rings` from pole to pole.
    pub fn ellipsoid(center: Vec3, radii: Vec3, segments: u32, rings: u32) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);
        let mut mesh = Self::default();
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let theta = v * std::f32::consts::PI;
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let phi = u * std::f32::consts::TAU;
                let unit = Vec3::new(theta.sin() * phi.sin(), theta.cos(), theta.sin() * phi.cos());
                mesh.positions.push(center + unit * radii);
                // The normal of an ellipsoid is the gradient: unit / radii.
                mesh.normals.push((unit / radii).normalize_or(unit));
                mesh.uvs.push(Vec2::new(u, v));
            }
        }
        let row = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let a = r * row + s;
                let b = a + row;
                mesh.indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        mesh
    }

    /// A sphere.
    pub fn sphere(center: Vec3, radius: f32, segments: u32, rings: u32) -> Self {
        Self::ellipsoid(center, Vec3::splat(radius), segments, rings)
    }

    /// A horizontal square of side `size` centred at `center`, facing +Y, split into `divisions`²
    /// quads.
    pub fn ground(center: Vec3, size: f32, divisions: u32) -> Self {
        let d = divisions.max(1);
        let mut mesh = Self::default();
        for z in 0..=d {
            for x in 0..=d {
                let u = x as f32 / d as f32;
                let v = z as f32 / d as f32;
                mesh.positions.push(center + Vec3::new((u - 0.5) * size, 0.0, (v - 0.5) * size));
                mesh.normals.push(Vec3::Y);
                mesh.uvs.push(Vec2::new(u, v));
            }
        }
        let row = d + 1;
        for z in 0..d {
            for x in 0..d {
                let a = z * row + x;
                let b = a + row;
                mesh.indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        mesh
    }

    /// An open cylinder along Y from `bottom` with `radius` and `height`.
    pub fn cylinder(bottom: Vec3, radius: f32, height: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let mut mesh = Self::default();
        for y in 0..=1_u32 {
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let phi = u * std::f32::consts::TAU;
                let n = Vec3::new(phi.sin(), 0.0, phi.cos());
                mesh.positions.push(bottom + n * radius + Vec3::Y * (height * y as f32));
                mesh.normals.push(n);
                mesh.uvs.push(Vec2::new(u, 1.0 - y as f32));
            }
        }
        let row = segments + 1;
        for s in 0..segments {
            let a = s;
            let b = a + row;
            mesh.indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
        mesh
    }

    /// A mesh from polygons (each a list of corner indices into `positions`, counter-clockwise
    /// from the outside), fan-triangulated, with smooth area-weighted vertex normals and zero
    /// uvs: how the template generator's surfaces (`humanart_model::generate`) are drawn.
    /// Polygons with fewer than 3 corners or out-of-range corners are an error.
    pub fn from_polygons<'a>(
        positions: &[Vec3],
        polygons: impl IntoIterator<Item = &'a [u32]>,
    ) -> Result<Self, RenderError> {
        let n = positions.len();
        if n > MAX_VERTICES {
            return Err(RenderError::TooLarge { what: "vertices", value: n as u64, limit: MAX_VERTICES as u64 });
        }
        let mut indices = Vec::new();
        for polygon in polygons {
            if polygon.len() < 3 || polygon.iter().any(|&i| i as usize >= n) {
                return Err(RenderError::InvalidMesh { what: "a polygon with fewer than 3 or out-of-range corners" });
            }
            for k in 1..polygon.len() - 1 {
                indices.extend_from_slice(&[polygon[0], polygon[k], polygon[k + 1]]);
            }
            if indices.len() > MAX_INDICES {
                return Err(RenderError::TooLarge {
                    what: "indices",
                    value: indices.len() as u64,
                    limit: MAX_INDICES as u64,
                });
            }
        }
        let mut normals = vec![Vec3::ZERO; n];
        for t in indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            // The cross product's length is twice the area: area weighting for free.
            let face = (b - a).cross(c - a);
            for &i in t {
                normals[i as usize] += face;
            }
        }
        let normals = normals.into_iter().map(|v| v.normalize_or(Vec3::Y)).collect();
        let mesh = Self { positions: positions.to_vec(), normals, uvs: vec![Vec2::ZERO; n], indices };
        mesh.validate()?;
        Ok(mesh)
    }

    /// A template mesh (`humanart_model::mesh::Mesh`) as a render mesh: the faces of the part named
    /// `part` (every face for `None`) triangulated as the template does, smooth area-weighted
    /// normals shared across UV seams (the template's normal keys, so hard edges stay hard), and
    /// texture coordinates `(u + tile, v)` with the face's UV tile in the integer part of u, as
    /// surfaced skin reads them (`crate::surface`). Every template vertex is kept (unused ones
    /// cost a few bytes), so vertex indices match the template's.
    ///
    /// # Errors
    /// [`RenderError::InvalidMesh`] for an unknown part; the size limits of [`Mesh::validate`].
    pub fn from_template(mesh: &humanart_model::mesh::Mesh, part: Option<&str>) -> Result<Self, RenderError> {
        let n = mesh.positions().len();
        if n > MAX_VERTICES {
            return Err(RenderError::TooLarge { what: "vertices", value: n as u64, limit: MAX_VERTICES as u64 });
        }
        let faces = match part {
            Some(name) => {
                let p = mesh
                    .parts()
                    .iter()
                    .find(|p| p.name == name)
                    .ok_or(RenderError::InvalidMesh { what: "no template part with that name" })?;
                p.first_face..p.first_face.saturating_add(p.face_count)
            }
            None => 0..mesh.face_count() as u32,
        };
        let triangles = mesh.triangulate();
        let (positions, keys, tiles) = (mesh.positions(), mesh.normal_keys(), mesh.face_tiles());
        let mut indices = Vec::new();
        let mut tile_of = vec![0_u16; n];
        let mut normals = vec![Vec3::ZERO; n];
        for (t, &face) in triangles.triangle_faces.iter().enumerate() {
            if !faces.contains(&face) {
                continue;
            }
            let Some(corners) = triangles.triangle(t) else { continue };
            if corners.iter().any(|&i| i as usize >= n) {
                return Err(RenderError::InvalidMesh { what: "a template triangle with out-of-range corners" });
            }
            let [a, b, c] = corners.map(|i| positions[i as usize]);
            let area = (b - a).cross(c - a);
            let tile = tiles.get(face as usize).copied().unwrap_or(0);
            for &i in &corners {
                let key = keys.get(i as usize).map_or(i as usize, |&k| k as usize).min(n - 1);
                normals[key] += area;
                tile_of[i as usize] = tile;
            }
            indices.extend_from_slice(&corners);
            if indices.len() > MAX_INDICES {
                return Err(RenderError::TooLarge {
                    what: "indices",
                    value: indices.len() as u64,
                    limit: MAX_INDICES as u64,
                });
            }
        }
        let normals = (0..n)
            .map(|i| {
                let key = keys.get(i).map_or(i, |&k| k as usize).min(n - 1);
                normals[key].normalize_or(Vec3::Y)
            })
            .collect();
        let uvs = mesh.uvs().iter().zip(&tile_of).map(|(uv, &tile)| Vec2::new(uv.x + f32::from(tile), uv.y)).collect();
        let out = Self { positions: positions.to_vec(), normals, uvs, indices };
        out.validate()?;
        Ok(out)
    }

    /// Appends `other` (indices rebased).
    pub fn append(&mut self, other: &Self) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// A head-sized bust from simple shapes (head, nose, ears, neck, shoulders), the head centre at
    /// `head_center`, facing +Z: a stand-in until the template generator (WP-06) is available.
    pub fn bust(head_center: Vec3) -> Self {
        let c = head_center;
        let mut mesh = Self::ellipsoid(c, Vec3::new(0.075, 0.105, 0.092), 64, 48);
        mesh.append(&Self::ellipsoid(c + Vec3::new(0.0, -0.01, 0.088), Vec3::new(0.014, 0.024, 0.018), 24, 16));
        mesh.append(&Self::ellipsoid(c + Vec3::new(0.0, -0.075, 0.03), Vec3::new(0.05, 0.035, 0.06), 32, 24));
        for side in [-1.0_f32, 1.0] {
            mesh.append(&Self::ellipsoid(
                c + Vec3::new(side * 0.074, 0.0, -0.005),
                Vec3::new(0.012, 0.03, 0.018),
                24,
                16,
            ));
            // Brow ridge.
            mesh.append(&Self::ellipsoid(
                c + Vec3::new(side * 0.03, 0.03, 0.078),
                Vec3::new(0.03, 0.012, 0.016),
                24,
                16,
            ));
        }
        mesh.append(&Self::cylinder(c + Vec3::new(0.0, -0.21, -0.01), 0.05, 0.13, 48));
        mesh.append(&Self::ellipsoid(c + Vec3::new(0.0, -0.27, -0.01), Vec3::new(0.2, 0.08, 0.1), 64, 32));
        mesh
    }
}

/// A plain PBR material (the skin, eye and hair models of later work packages bring their own).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Linear Rec.709 base colour.
    pub base_color: [f32; 3],
    /// Perceptual roughness, 0..1.
    pub roughness: f32,
    /// Metalness, 0..1.
    pub metallic: f32,
    /// Reflectance at normal incidence of the dielectric part (0.04 for most, 0.028 for skin).
    pub f0: f32,
    /// Normalised thickness for the subsurface passes, 0..1.
    pub thickness: f32,
    /// Material class (0 standard; reserved: 1 skin, 2 eye, 3 hair, 4 teeth).
    pub class: u8,
}

impl Default for Material {
    fn default() -> Self {
        Self { base_color: [0.5; 3], roughness: 0.5, metallic: 0.0, f0: 0.04, thickness: 0.0, class: 0 }
    }
}

impl Material {
    /// Checks the ranges.
    pub fn validate(&self) -> Result<(), RenderError> {
        let in01 = |v: f32| (0.0..=1.0).contains(&v);
        if !self.base_color.iter().all(|&c| c.is_finite() && c >= 0.0)
            || !in01(self.roughness)
            || !in01(self.metallic)
            || !in01(self.f0)
            || !in01(self.thickness)
        {
            return Err(RenderError::InvalidParameter { what: "material values outside their ranges" });
        }
        Ok(())
    }
}

/// One drawable: a mesh of the scene with a transform and a material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Object {
    /// Index into [`Scene::meshes`].
    pub mesh: usize,
    /// Object to world.
    pub transform: Mat4,
    /// Its material.
    pub material: Material,
    /// The id the picking buffer reports (0 is reserved for "nothing").
    pub pick_id: u32,
    /// Drawn with the selection highlight.
    pub selected: bool,
    /// Casts shadows.
    pub casts_shadow: bool,
    /// The skin parameters, read when the material's class is [`crate::sss::SKIN_CLASS`].
    pub skin: SkinShading,
}

/// Meshes and the objects that draw them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    /// The meshes.
    pub meshes: Vec<Mesh>,
    /// The objects.
    pub objects: Vec<Object>,
}

impl Scene {
    /// Adds a mesh and an object drawing it; returns the object index.
    pub fn add(&mut self, mesh: Mesh, material: Material) -> usize {
        self.meshes.push(mesh);
        let pick_id = self.objects.len() as u32 + 1;
        self.objects.push(Object {
            mesh: self.meshes.len() - 1,
            transform: Mat4::IDENTITY,
            material,
            pick_id,
            selected: false,
            casts_shadow: true,
            skin: SkinShading::default(),
        });
        self.objects.len() - 1
    }

    /// An object that draws nothing (the placeholder row of an empty object table).
    fn placeholder_object() -> Object {
        Object {
            mesh: 0,
            transform: Mat4::IDENTITY,
            material: Material::default(),
            pick_id: 0,
            selected: false,
            casts_shadow: false,
            skin: SkinShading::default(),
        }
    }

    /// Checks every mesh, material and reference, and the limits.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.objects.len() > MAX_OBJECTS {
            return Err(RenderError::TooLarge {
                what: "objects",
                value: self.objects.len() as u64,
                limit: MAX_OBJECTS as u64,
            });
        }
        let (mut vertices, mut indices) = (0_usize, 0_usize);
        for mesh in &self.meshes {
            mesh.validate()?;
            vertices += mesh.positions.len();
            indices += mesh.indices.len();
        }
        if vertices > MAX_VERTICES || indices > MAX_INDICES {
            return Err(RenderError::TooLarge {
                what: "scene vertices",
                value: vertices as u64,
                limit: MAX_VERTICES as u64,
            });
        }
        for object in &self.objects {
            if object.mesh >= self.meshes.len() {
                return Err(RenderError::InvalidMesh { what: "an object names a mesh that does not exist" });
            }
            if !object.transform.is_finite() {
                return Err(RenderError::InvalidParameter { what: "an object transform is not finite" });
            }
            object.material.validate()?;
            object.skin.validate()?;
        }
        Ok(())
    }

    /// The world bounding box of every object.
    pub fn bounds(&self) -> Aabb {
        let mut out = Aabb::EMPTY;
        for object in &self.objects {
            let Some(mesh) = self.meshes.get(object.mesh) else { continue };
            for p in &mesh.positions {
                out = out.extend(object.transform.transform_point3(*p));
            }
        }
        out
    }

    /// The bounding sphere of the objects that cast shadows (what shadow maps are fitted to).
    pub fn shadow_bounds(&self) -> Sphere {
        let mut bounds = Aabb::EMPTY;
        for object in self.objects.iter().filter(|o| o.casts_shadow) {
            let Some(mesh) = self.meshes.get(object.mesh) else { continue };
            for p in &mesh.positions {
                bounds = bounds.extend(object.transform.transform_point3(*p));
            }
        }
        if bounds.is_empty() {
            return Sphere { center: Vec3::ZERO, radius: 1.0 };
        }
        Sphere { center: bounds.center(), radius: bounds.half_extents().length().max(1e-3) }
    }

    /// Whether every object that only receives shadows lies inside the box around the casters (to
    /// a centimetre): a tear line on an eye does, a ground does not. Only then can a shadow map be
    /// narrowed to the casters the camera sees: a receiver elsewhere may be in view, far from
    /// them, in the shadow of a caster outside the view.
    pub fn receivers_within_casters(&self) -> bool {
        let mut casters = Aabb::EMPTY;
        let mut receivers = Aabb::EMPTY;
        for object in &self.objects {
            let Some(mesh) = self.meshes.get(object.mesh) else { continue };
            for p in &mesh.positions {
                let p = object.transform.transform_point3(*p);
                if object.casts_shadow {
                    casters = casters.extend(p);
                } else {
                    receivers = receivers.extend(p);
                }
            }
        }
        if receivers.is_empty() {
            return true;
        }
        let slack = Vec3::splat(0.01);
        let (lo, hi) = (casters.min - slack, casters.max + slack);
        !casters.is_empty()
            && receivers.min.x >= lo.x
            && receivers.min.y >= lo.y
            && receivers.min.z >= lo.z
            && receivers.max.x <= hi.x
            && receivers.max.y <= hi.y
            && receivers.max.z <= hi.z
    }

    /// The shadow casters as boxes per band of height ([`crate::focus`]): what a shadow map is
    /// narrowed to when the camera sees only part of them.
    pub fn shadow_slabs(&self) -> CasterSlabs {
        let mut bounds = Aabb::EMPTY;
        let casters =
            || self.objects.iter().filter(|o| o.casts_shadow).filter_map(|o| self.meshes.get(o.mesh).map(|m| (o, m)));
        for (object, mesh) in casters() {
            for p in &mesh.positions {
                bounds = bounds.extend(object.transform.transform_point3(*p));
            }
        }
        if bounds.is_empty() {
            return CasterSlabs::EMPTY;
        }
        let mut slabs = CasterSlabs::new(bounds.min.y, bounds.max.y);
        for (object, mesh) in casters() {
            for p in &mesh.positions {
                slabs.add(object.transform.transform_point3(*p));
            }
        }
        slabs
    }

    /// The material test scene: a ground plane, two rows of five spheres (dielectric and metal,
    /// roughness 0.1 to 1.0) and a head-sized bust in the middle, around the origin, head at 1.6 m.
    pub fn material_test() -> Self {
        let mut scene = Self::default();
        let ground = scene.add(
            Mesh::ground(Vec3::new(0.0, 1.2, 0.0), 3.0, 8),
            Material { base_color: [0.3, 0.3, 0.3], roughness: 0.8, ..Material::default() },
        );
        scene.objects[ground].casts_shadow = false;
        for row in 0..2 {
            for i in 0..5 {
                let roughness = 0.1 + 0.225 * i as f32;
                let x = (i as f32 - 2.0) * 0.12;
                let y = 1.28 + row as f32 * 0.12;
                let material = if row == 0 {
                    Material { base_color: [0.8, 0.12, 0.08], roughness, ..Material::default() }
                } else {
                    Material { base_color: [0.95, 0.75, 0.4], roughness, metallic: 1.0, ..Material::default() }
                };
                scene.add(Mesh::sphere(Vec3::new(x, y, 0.25), 0.05, 48, 32), material);
            }
        }
        scene.add(
            Mesh::bust(Vec3::new(0.0, 1.6, -0.05)),
            Material {
                base_color: [0.62, 0.45, 0.37],
                roughness: 0.45,
                f0: 0.028,
                thickness: 0.5,
                ..Material::default()
            },
        );
        scene
    }
}

/// Where one mesh lives in the shared streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRange {
    /// First vertex.
    pub base_vertex: u32,
    /// Vertices.
    pub vertex_count: u32,
    /// First index.
    pub first_index: u32,
    /// Indices.
    pub index_count: u32,
}

impl MeshRange {
    /// No vertices, no indices.
    pub const EMPTY: Self = Self { base_vertex: 0, vertex_count: 0, first_index: 0, index_count: 0 };
}

/// One draw: an object with its index range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draw {
    /// The object (the instance index the shaders read the object table with).
    pub object: u32,
    /// The mesh (which group-1 bind group the draw needs when the mesh is deformed).
    pub mesh: u32,
    /// First index.
    pub first_index: u32,
    /// Indices.
    pub index_count: u32,
    /// Added to every index.
    pub base_vertex: i32,
    /// Casts shadows.
    pub casts_shadow: bool,
    /// Selected.
    pub selected: bool,
}

humanart_gpu::pod_struct! {
    /// One object as the shaders read it (`struct Object` in `shaders/render/scene.wgsl`).
    #[derive(Debug, PartialEq)]
    pub struct GpuObject(size = 256, align = 4) {
        /// Object to world.
        pub model: [f32; 16],
        /// The previous frame's object to world.
        pub prev_model: [f32; 16],
        /// Inverse transpose of `model`.
        pub normal_matrix: [f32; 16],
        /// Linear base colour, roughness.
        pub base_color_roughness: [f32; 4],
        /// Metalness, f0, thickness, class.
        pub material: [f32; 4],
        /// Skin: secondary roughness, lobe mix, scatter scale, translucency ([`SkinShading::to_gpu`]).
        pub skin: [f32; 4],
        /// Pick id, flags, first index, base vertex (what the passes that pull triangles through
        /// the index buffer add to every index, like `draw_indexed` does).
        pub ids: [u32; 4],
    }
}

impl GpuObject {
    /// Packs `object` with its index range.
    pub fn new(object: &Object, range: &MeshRange, prev_model: Mat4) -> Self {
        let m = &object.material;
        Self {
            model: object.transform.to_cols_array(),
            prev_model: prev_model.to_cols_array(),
            normal_matrix: Mat4::from_mat3(object.transform.normal_matrix()).to_cols_array(),
            base_color_roughness: [m.base_color[0], m.base_color[1], m.base_color[2], m.roughness],
            material: [m.metallic, m.f0, m.thickness, f32::from(m.class)],
            skin: object.skin.to_gpu(),
            ids: [
                object.pick_id,
                u32::from(object.selected) | (u32::from(object.skin.surface && m.class == SKIN_CLASS) << 1),
                range.first_index,
                range.base_vertex,
            ],
        }
    }
}

/// The streams of a deformed mesh `[EVAL-05]`: the deformer's output buffers (packed f32 triples,
/// the mesh's vertices only, base vertex 0) and a copy of the mesh's texture coordinates.
#[derive(Debug)]
pub struct ExternalStreams {
    /// Deformed positions.
    pub positions: wgpu::Buffer,
    /// The positions before the last change (the deformer keeps them).
    pub prev_positions: wgpu::Buffer,
    /// Deformed unit normals.
    pub normals: wgpu::Buffer,
    /// The mesh's texture coordinates from its first vertex (owned by the scene).
    pub uvs: BufferId,
}

/// The range the draws of mesh `mesh` use: base vertex 0 when its streams are a deformer's.
fn draw_range(ranges: &[MeshRange], external: &[Option<ExternalStreams>], mesh: usize) -> MeshRange {
    let range = ranges.get(mesh).copied().unwrap_or(MeshRange::EMPTY);
    if external.get(mesh).is_some_and(Option::is_some) { MeshRange { base_vertex: 0, ..range } } else { range }
}

/// The scene on the GPU. See the module documentation.
#[derive(Debug)]
pub struct GpuScene {
    positions: BufferId,
    prev_positions: BufferId,
    staging: BufferId,
    normals: BufferId,
    uvs: BufferId,
    indices: BufferId,
    objects: BufferId,
    ranges: Vec<MeshRange>,
    draws: Vec<Draw>,
    cpu_objects: Vec<Object>,
    gpu_objects: Vec<GpuObject>,
    last_models: Vec<Mat4>,
    /// Per mesh: new positions are in the staging stream.
    pending: Vec<bool>,
    /// Per mesh: positions changed in the previous frame.
    moved: Vec<bool>,
    objects_dirty: bool,
    /// Per mesh: the deformer's streams when the mesh is deformed on the GPU.
    external: Vec<Option<ExternalStreams>>,
    /// Bumped whenever streams are attached or detached (the renderer rebuilds its bind groups).
    stream_generation: u64,
    bounds: Sphere,
    slabs: CasterSlabs,
    focusable: bool,
    vertex_count: u32,
    index_count: u32,
}

/// The bind-group layout entries of the scene streams (group 1).
pub const SCENE_LAYOUT: [wgpu::BindGroupLayoutEntry; 6] = {
    const fn storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
    [storage(0), storage(1), storage(2), storage(3), storage(4), storage(5)]
};

impl GpuScene {
    /// Uploads `scene` into new buffers in `resources`.
    pub fn new(resources: &mut Resources, scene: &Scene) -> Result<Self, RenderError> {
        scene.validate()?;
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut uvs: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut ranges = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            ranges.push(MeshRange {
                base_vertex: (positions.len() / 3) as u32,
                vertex_count: mesh.positions.len() as u32,
                first_index: indices.len() as u32,
                index_count: mesh.indices.len() as u32,
            });
            positions.extend(mesh.positions.iter().flat_map(|p| p.to_array()));
            normals.extend(mesh.normals.iter().flat_map(|n| n.to_array()));
            uvs.extend(mesh.uvs.iter().flat_map(|u| u.to_array()));
            indices.extend_from_slice(&mesh.indices);
        }
        // Empty streams still need a binding: at least one element each.
        for stream in [&mut positions, &mut normals, &mut uvs] {
            if stream.is_empty() {
                stream.extend_from_slice(&[0.0; 4]);
            }
        }
        if indices.is_empty() {
            indices.extend_from_slice(&[0; 4]);
        }
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut make =
            |bytes: &[u8], usage: wgpu::BufferUsages, purpose: &'static str| -> Result<BufferId, RenderError> {
                let (id, _) = resources.create_buffer(&BufferDesc {
                    size: bytes.len() as u64,
                    usage,
                    tag: Tag::shared(Category::Geometry, purpose),
                })?;
                resources.write_buffer(id, 0, bytes)?;
                Ok(id)
            };
        let position_bytes = slice_as_bytes(&positions);
        let positions_id = make(position_bytes, storage | wgpu::BufferUsages::COPY_SRC, "scene positions")?;
        let prev_positions = make(position_bytes, storage, "scene previous positions")?;
        let staging = make(
            position_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "scene position staging",
        )?;
        let normals_id = make(slice_as_bytes(&normals), storage, "scene normals")?;
        let uvs_id = make(slice_as_bytes(&uvs), storage | wgpu::BufferUsages::COPY_SRC, "scene uvs")?;
        let indices_id = make(slice_as_bytes(&indices), storage | wgpu::BufferUsages::INDEX, "scene indices")?;
        let mut gpu_objects = Vec::with_capacity(scene.objects.len().max(1));
        let mut draws = Vec::with_capacity(scene.objects.len());
        for (i, object) in scene.objects.iter().enumerate() {
            let range = ranges.get(object.mesh).copied().ok_or(RenderError::InvalidMesh { what: "object mesh" })?;
            gpu_objects.push(GpuObject::new(object, &range, object.transform));
            draws.push(Draw {
                object: i as u32,
                mesh: object.mesh as u32,
                first_index: range.first_index,
                index_count: range.index_count,
                base_vertex: range.base_vertex as i32,
                casts_shadow: object.casts_shadow,
                selected: object.selected,
            });
        }
        let empty = [GpuObject::new(&Scene::placeholder_object(), &MeshRange::EMPTY, Mat4::IDENTITY)];
        let object_bytes = if gpu_objects.is_empty() { slice_as_bytes(&empty) } else { slice_as_bytes(&gpu_objects) };
        let objects = make(object_bytes, storage, "scene objects")?;
        Ok(Self {
            positions: positions_id,
            prev_positions,
            staging,
            normals: normals_id,
            uvs: uvs_id,
            indices: indices_id,
            objects,
            pending: vec![false; ranges.len()],
            moved: vec![false; ranges.len()],
            external: (0..ranges.len()).map(|_| None).collect(),
            stream_generation: 0,
            last_models: scene.objects.iter().map(|o| o.transform).collect(),
            cpu_objects: scene.objects.clone(),
            ranges,
            draws,
            gpu_objects,
            objects_dirty: false,
            bounds: scene.shadow_bounds(),
            slabs: scene.shadow_slabs(),
            focusable: scene.receivers_within_casters(),
            vertex_count: (positions.len() / 3) as u32,
            index_count: indices.len() as u32,
        })
    }

    /// The bounding sphere of the shadow casters at upload (what shadow maps are fitted to).
    pub fn bounds(&self) -> Sphere {
        self.bounds
    }

    /// The shadow casters at upload as boxes per band of height ([`crate::focus`]).
    pub fn caster_slabs(&self) -> &CasterSlabs {
        &self.slabs
    }

    /// Whether a shadow map may be narrowed to the casters the camera sees ([`crate::focus`]):
    /// see [`Scene::receivers_within_casters`].
    pub fn shadow_focusable(&self) -> bool {
        self.focusable
    }

    /// The draws, one per object.
    pub fn draws(&self) -> &[Draw] {
        &self.draws
    }

    /// Where each mesh lives in the streams.
    pub fn ranges(&self) -> &[MeshRange] {
        &self.ranges
    }

    /// The index buffer.
    pub fn index_buffer(&self) -> BufferId {
        self.indices
    }

    /// Vertices and indices uploaded.
    pub fn counts(&self) -> (u32, u32) {
        (self.vertex_count, self.index_count)
    }

    /// The buffers of the group-1 bind group, in binding order.
    pub fn bind_buffers(&self) -> [BufferId; 6] {
        [self.positions, self.prev_positions, self.normals, self.uvs, self.objects, self.indices]
    }

    /// The deformer's streams of mesh `mesh`, when it is deformed on the GPU.
    pub fn external_streams(&self, mesh: usize) -> Option<&ExternalStreams> {
        self.external.get(mesh).and_then(Option::as_ref)
    }

    /// A counter that changes whenever streams are attached or detached.
    pub fn stream_generation(&self) -> u64 {
        self.stream_generation
    }

    /// The range a mesh's draws use: base vertex 0 when its streams are the deformer's.
    fn draw_range(&self, mesh: usize) -> MeshRange {
        draw_range(&self.ranges, &self.external, mesh)
    }

    /// Re-derives the draws and the object table after a mesh's streams changed hands.
    fn rebase_draws(&mut self, mesh: usize) {
        let range = self.draw_range(mesh);
        for d in self.draws.iter_mut().filter(|d| d.mesh as usize == mesh) {
            d.base_vertex = range.base_vertex as i32;
        }
        self.objects_dirty = true;
        self.stream_generation = self.stream_generation.wrapping_add(1);
    }

    /// Hands mesh `mesh` over to `deformer` `[EVAL-05]`: from the next frame every pass pulls its
    /// positions, previous positions and normals from the deformer's output buffers (no copy) and
    /// the scene stops keeping its previous positions. Records the one-time copy of the mesh's
    /// texture coordinates into `encoder`. The deformer's vertex order must be the mesh's (a
    /// template uploaded in its own vertex order) and its vertex count equal. Detach the mesh
    /// ([`Self::detach_deformer`]) before destroying the deformer.
    pub fn attach_deformer(
        &mut self,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
        mesh: usize,
        deformer: &GpuDeformer,
    ) -> Result<(), RenderError> {
        let range = *self.ranges.get(mesh).ok_or(RenderError::InvalidParameter { what: "mesh index" })?;
        if deformer.vertex_count() != range.vertex_count || range.vertex_count == 0 {
            return Err(RenderError::InvalidMesh { what: "the deformer's vertex count differs from the mesh's" });
        }
        let outputs = deformer.outputs();
        let bytes = u64::from(range.vertex_count) * 12;
        if [outputs.positions, outputs.prev_positions, outputs.normals].iter().any(|b| b.size() < bytes) {
            return Err(RenderError::InvalidMesh { what: "a deformer output buffer is smaller than the mesh" });
        }
        self.detach_deformer(resources, mesh);
        let uv_bytes = u64::from(range.vertex_count) * 8;
        let (uvs, _) = resources.create_buffer(&BufferDesc {
            size: uv_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            tag: Tag::shared(Category::Geometry, "deformed mesh uvs"),
        })?;
        let stale = || RenderError::Gpu(humanart_gpu::GpuError::Stale { what: "scene buffer" });
        let source = resources.buffer(self.uvs).ok_or_else(stale)?.raw();
        let target = resources.buffer(uvs).ok_or_else(stale)?.raw();
        encoder.copy_buffer_to_buffer(source, u64::from(range.base_vertex) * 8, target, 0, uv_bytes);
        if let Some(slot) = self.external.get_mut(mesh) {
            *slot = Some(ExternalStreams {
                positions: outputs.positions.clone(),
                prev_positions: outputs.prev_positions.clone(),
                normals: outputs.normals.clone(),
                uvs,
            });
        }
        if let Some(p) = self.pending.get_mut(mesh) {
            *p = false;
        }
        if let Some(m) = self.moved.get_mut(mesh) {
            *m = false;
        }
        self.rebase_draws(mesh);
        Ok(())
    }

    /// Gives mesh `mesh` its own streams back (the positions last uploaded for it); nothing
    /// happens when it is not deformed.
    pub fn detach_deformer(&mut self, resources: &mut Resources, mesh: usize) {
        if let Some(streams) = self.external.get_mut(mesh).and_then(Option::take) {
            let _ = resources.destroy_buffer(streams.uvs);
            self.rebase_draws(mesh);
        }
    }

    /// Moves an object; its motion vectors follow from the previous transform.
    pub fn set_transform(&mut self, object: usize, transform: Mat4) -> Result<(), RenderError> {
        if !transform.is_finite() {
            return Err(RenderError::InvalidParameter { what: "an object transform is not finite" });
        }
        let o = self.cpu_objects.get_mut(object).ok_or(RenderError::InvalidParameter { what: "object index" })?;
        o.transform = transform;
        self.objects_dirty = true;
        Ok(())
    }

    /// Changes an object's skin parameters (read when its material class is skin).
    pub fn set_skin(&mut self, object: usize, skin: SkinShading) -> Result<(), RenderError> {
        skin.validate()?;
        let o = self.cpu_objects.get_mut(object).ok_or(RenderError::InvalidParameter { what: "object index" })?;
        o.skin = skin;
        self.objects_dirty = true;
        Ok(())
    }

    /// Marks an object selected (or not) for the selection overlay.
    pub fn set_selected(&mut self, object: usize, selected: bool) -> Result<(), RenderError> {
        let o = self.cpu_objects.get_mut(object).ok_or(RenderError::InvalidParameter { what: "object index" })?;
        o.selected = selected;
        if let Some(d) = self.draws.get_mut(object) {
            d.selected = selected;
        }
        self.objects_dirty = true;
        Ok(())
    }

    /// New positions of mesh `mesh` (a deformation), effective in the next encoded frame. The
    /// count must equal the mesh's vertex count. No allocation.
    pub fn set_positions(
        &mut self,
        resources: &mut Resources,
        mesh: usize,
        positions: &[Vec3],
    ) -> Result<(), RenderError> {
        let range = *self.ranges.get(mesh).ok_or(RenderError::InvalidParameter { what: "mesh index" })?;
        if self.external_streams(mesh).is_some() {
            return Err(RenderError::InvalidParameter { what: "the mesh is deformed on the GPU (detach it first)" });
        }
        if positions.len() != range.vertex_count as usize {
            return Err(RenderError::InvalidMesh { what: "position count differs from the mesh's vertex count" });
        }
        if !positions.iter().all(|p| p.is_finite()) {
            return Err(RenderError::InvalidMesh { what: "a position is not finite" });
        }
        let offset = u64::from(range.base_vertex) * 12;
        // Vec3 is three packed f32; written one vertex at a time to stay allocation-free.
        for (i, p) in positions.iter().enumerate() {
            resources.write_buffer(self.staging, offset + i as u64 * 12, slice_as_bytes(&p.to_array()))?;
        }
        if let Some(flag) = self.pending.get_mut(mesh) {
            *flag = true;
        }
        Ok(())
    }

    /// Records the start of a frame: keeps the previous positions and transforms, moves new
    /// positions from staging into place, uploads the object table if anything changed.
    pub fn begin_frame(
        &mut self,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), RenderError> {
        let buffer = |id: BufferId| {
            resources
                .buffer(id)
                .map(|b| b.raw().clone())
                .ok_or(RenderError::Gpu(humanart_gpu::GpuError::Stale { what: "scene buffer" }))
        };
        // A deformed mesh never has its flags set (attach clears them, set_positions refuses it):
        // the deformer keeps its previous positions, no copy happens here.
        if self.pending.iter().chain(&self.moved).any(|&f| f) {
            let (current, previous, staging) =
                (buffer(self.positions)?, buffer(self.prev_positions)?, buffer(self.staging)?);
            for (i, range) in self.ranges.iter().enumerate() {
                let pending = self.pending.get(i).copied().unwrap_or(false);
                let moved = self.moved.get(i).copied().unwrap_or(false);
                let offset = u64::from(range.base_vertex) * 12;
                let size = u64::from(range.vertex_count) * 12;
                if (pending || moved) && size > 0 {
                    encoder.copy_buffer_to_buffer(&current, offset, &previous, offset, size);
                }
                if pending && size > 0 {
                    encoder.copy_buffer_to_buffer(&staging, offset, &current, offset, size);
                }
                if let Some(m) = self.moved.get_mut(i) {
                    *m = pending;
                }
                if let Some(p) = self.pending.get_mut(i) {
                    *p = false;
                }
            }
        }
        // Object table: prev_model = last frame's model.
        let mut changed = self.objects_dirty;
        for ((gpu, object), last) in self.gpu_objects.iter_mut().zip(&self.cpu_objects).zip(self.last_models.iter_mut())
        {
            if gpu.prev_model != last.to_cols_array() || *last != object.transform || self.objects_dirty {
                let range = draw_range(&self.ranges, &self.external, object.mesh);
                *gpu = GpuObject::new(object, &range, *last);
                *last = object.transform;
                changed = true;
            }
        }
        if changed && !self.gpu_objects.is_empty() {
            resources.write_buffer(self.objects, 0, slice_as_bytes(&self.gpu_objects))?;
        }
        self.objects_dirty = false;
        Ok(())
    }

    /// Frees the buffers (the deformers' buffers stay theirs).
    pub fn destroy(&self, resources: &mut Resources) {
        for streams in self.external.iter().flatten() {
            let _ = resources.destroy_buffer(streams.uvs);
        }
        for id in
            [self.positions, self.prev_positions, self.staging, self.normals, self.uvs, self.indices, self.objects]
        {
            let _ = resources.destroy_buffer(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_meshes_are_valid_and_closed_where_expected() {
        let s = Mesh::sphere(Vec3::ZERO, 1.0, 16, 8);
        s.validate().unwrap();
        assert!(s.positions.iter().all(|p| (p.length() - 1.0).abs() < 1e-5));
        assert!(s.positions.iter().zip(&s.normals).all(|(p, n)| (n.dot(*p) - 1.0).abs() < 1e-5));
        Mesh::ground(Vec3::ZERO, 2.0, 4).validate().unwrap();
        Mesh::cylinder(Vec3::ZERO, 0.5, 1.0, 12).validate().unwrap();
        let bust = Mesh::bust(Vec3::new(0.0, 1.6, 0.0));
        bust.validate().unwrap();
        let b = bust.bounds();
        assert!(b.size().y > 0.3 && b.size().y < 0.6, "head-sized: {:?}", b.size());
    }

    /// Winding: counter-clockwise from the outside, so the geometric normal agrees with the
    /// stored normal.
    #[test]
    fn triangles_face_outwards() {
        for mesh in [
            Mesh::sphere(Vec3::ZERO, 1.0, 16, 8),
            Mesh::ground(Vec3::ZERO, 1.0, 2),
            Mesh::cylinder(Vec3::ZERO, 1.0, 1.0, 8),
        ] {
            for t in mesh.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
                let n = (b - a).cross(c - a);
                if n.length() < 1e-6 {
                    continue;
                }
                let stored = mesh.normals[t[0] as usize] + mesh.normals[t[1] as usize] + mesh.normals[t[2] as usize];
                assert!(n.dot(stored) > 0.0, "a triangle faces inwards");
            }
        }
    }

    #[test]
    fn polygons_triangulate_with_smooth_normals() {
        // A unit cube from quads, counter-clockwise from outside.
        let p: Vec<Vec3> =
            (0..8).map(|i| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32)).collect();
        let quads: [[u32; 4]; 6] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let mesh = Mesh::from_polygons(&p, quads.iter().map(|q| &q[..])).unwrap();
        assert_eq!(mesh.indices.len(), 36);
        let center = Vec3::splat(0.5);
        for (v, n) in mesh.positions.iter().zip(&mesh.normals) {
            assert!(n.dot(*v - center) > 0.0, "normals point outwards");
        }
        assert!(Mesh::from_polygons(&p, [&[0_u32, 1][..]]).is_err());
        assert!(Mesh::from_polygons(&p, [&[0_u32, 1, 99][..]]).is_err());
    }

    #[test]
    fn invalid_meshes_are_rejected() {
        let mut m = Mesh::sphere(Vec3::ZERO, 1.0, 8, 4);
        m.indices.push(9999);
        assert!(m.validate().is_err());
        let mut m = Mesh::sphere(Vec3::ZERO, 1.0, 8, 4);
        m.normals.pop();
        assert!(m.validate().is_err());
        let mut m = Mesh::sphere(Vec3::ZERO, 1.0, 8, 4);
        m.positions[0] = Vec3::NAN;
        assert!(m.validate().is_err());
        let mut scene = Scene::default();
        scene.add(Mesh::sphere(Vec3::ZERO, 1.0, 8, 4), Material { roughness: 2.0, ..Material::default() });
        assert!(scene.validate().is_err());
        let mut scene = Scene::material_test();
        scene.objects[0].mesh = 999;
        assert!(scene.validate().is_err());
    }

    #[test]
    fn material_test_scene_is_valid() {
        let scene = Scene::material_test();
        scene.validate().unwrap();
        assert_eq!(scene.objects.len(), 12);
        let ids: Vec<u32> = scene.objects.iter().map(|o| o.pick_id).collect();
        assert!(ids.iter().all(|&i| i > 0), "pick id 0 means nothing");
        let s = scene.shadow_bounds();
        assert!(s.radius > 0.1 && s.radius < 1.0, "the ground does not cast: {s:?}");
    }
}
