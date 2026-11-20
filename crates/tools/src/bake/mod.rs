//! The bakers `[BAKE-01]`..`[BAKE-08]`: maps in UV space computed from a mesh.
//!
//! Every baker has a CPU reference on small meshes `[BAKE-08]`; the ray-traced ones (thickness,
//! ambient occlusion, cavity, bent normals) and the seam-padding dilation also run on the GPU
//! ([`gpu`]) and are compared against the CPU reference. Everything is deterministic: ray
//! directions come from a fixed low-discrepancy sequence rotated by a hash of the texel
//! ([`sampling`]), never from a clock or a thread-dependent order.
//!
//! Module map:
//!
//! - [`BakeMesh`]: the mesh a baker reads: positions (rest or deformed), the template's vertex
//!   normals and tangents (`humanart_model::eval::normals`, the frames the renderer uses), UVs,
//!   triangles with their UV tile, vertex regions.
//! - [`BakeTarget`] and [`BakeMap`]: which tile at which size, and the float map a baker writes,
//!   with a per-texel coverage flag.
//! - [`raster`]: the UV-space rasteriser: which triangle covers each texel centre, and where.
//! - [`bvh`]: the bounding-volume hierarchy and ray casts shared by the ray-traced bakers.
//! - [`sampling`]: the deterministic ray directions (mirrored in `shaders/bake/common.wgsl`).
//! - [`thickness`]: thickness and translucency with a per-region scale `[BAKE-01]`.
//! - [`occlusion`]: ambient occlusion, cavity and bent normals `[BAKE-02]`.
//! - [`surface`]: mean curvature, tangent frames and region ids `[BAKE-03]`.
//! - [`dilate`]: seam-padding dilation `[BAKE-03]`, verified through the mip chain.
//! - [`detail`]: normal and height detail from a high-resolution source `[BAKE-04]`.
//! - [`tension`]: wrinkle and tension maps with fine-line fields `[BAKE-05]`.
//! - [`hair_card`]: the hair-card maps `[BAKE-06]`.
//! - [`strain`]: rest edge lengths and region strain normalisation tables `[BAKE-07]`.
//! - [`gpu`]: the GPU bakers.

pub mod bvh;
pub mod detail;
pub mod dilate;
pub mod gpu;
pub mod hair_card;
pub mod occlusion;
pub mod raster;
pub mod sampling;
pub mod strain;
pub mod surface;
pub mod tension;
pub mod thickness;

use std::fmt;

use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{Vec2, Vec3, Vec4};
use humanart_model::eval::normals::{TriangleFrames, normal_corner_gather, own_corner_gather};
use humanart_model::{Mesh, ModelError};

/// The largest map side a baker accepts, in texels.
pub const MAX_MAP_SIZE: u32 = 16384;
/// The most channels a [`BakeMap`] has.
pub const MAX_CHANNELS: u8 = 4;

/// Why a bake failed.
#[derive(Debug)]
pub enum BakeError {
    /// The request or its input is malformed.
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
    /// The mesh is not a valid template mesh.
    Model(ModelError),
    /// Converting a map to an image failed.
    Io(humanart_io::Error),
    /// The device refused a resource, a pipeline or a readback.
    Gpu(humanart_gpu::GpuError),
    /// A shader could not be composed.
    Shader(humanart_shaders::ShaderError),
}

impl BakeError {
    /// An [`BakeError::Invalid`] error.
    pub const fn invalid(what: &'static str) -> Self {
        Self::Invalid { what }
    }
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { what } => write!(f, "bake: {what}"),
            Self::Model(e) => write!(f, "bake: {e}"),
            Self::Io(e) => write!(f, "bake: {e}"),
            Self::Gpu(e) => write!(f, "bake: {e}"),
            Self::Shader(e) => write!(f, "bake shader: {e}"),
        }
    }
}

impl std::error::Error for BakeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid { .. } => None,
            Self::Model(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
        }
    }
}

impl From<ModelError> for BakeError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<humanart_io::Error> for BakeError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<humanart_gpu::GpuError> for BakeError {
    fn from(e: humanart_gpu::GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<humanart_shaders::ShaderError> for BakeError {
    fn from(e: humanart_shaders::ShaderError) -> Self {
        Self::Shader(e)
    }
}

/// The result type of the bakers.
pub type Result<T> = std::result::Result<T, BakeError>;

/// The region a vertex without a region map belongs to.
pub const NO_REGION: u16 = u16::MAX;

/// The mesh a baker reads, as triangles with structure-of-arrays vertex attributes `[PERF-R2]`.
///
/// Normals and tangents are computed from the given positions by the template's own normal and
/// tangent passes (`[EVAL-04]`, angle-weighted, smooth across UV seams), so a baked tangent-space
/// map is expressed in exactly the frame the renderer reconstructs. Positions may be the rest
/// shape or any deformed shape (`[BAKE-02]`: "from the deformed or rest mesh").
#[derive(Clone, Debug, PartialEq)]
pub struct BakeMesh {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    tangents: Vec<Vec4>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,
    triangle_tiles: Vec<u16>,
    triangle_normals: Vec<Vec3>,
    vertex_regions: Vec<u16>,
    position_keys: Vec<u32>,
    extent: f32,
}

impl BakeMesh {
    /// The bake view of `mesh` with the vertex positions `positions` (one per vertex: the rest
    /// positions `mesh.positions()` or a deformed shape) and optionally the region of every vertex
    /// (`RegionMap::vertex_regions`); without regions every vertex is in [`NO_REGION`].
    pub fn new(mesh: &Mesh, positions: &[Vec3], vertex_regions: Option<&[u16]>) -> Result<Self> {
        let n = mesh.vertex_count();
        if positions.len() != n {
            return Err(BakeError::invalid("the position count differs from the mesh's vertex count"));
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return Err(BakeError::invalid("a position is not finite"));
        }
        let vertex_regions = match vertex_regions {
            Some(r) if r.len() == n => r.to_vec(),
            Some(_) => return Err(BakeError::invalid("the region count differs from the mesh's vertex count")),
            None => vec![NO_REGION; n],
        };
        let triangulation = mesh.triangulate();
        let indices = triangulation.indices;
        let tiles = mesh.face_tiles();
        let triangle_tiles: Vec<u16> =
            triangulation.triangle_faces.iter().map(|&f| tiles.get(f as usize).copied().unwrap_or(0)).collect();
        let mut frames = TriangleFrames::new(triangle_tiles.len());
        frames.compute(positions, mesh.uvs(), &indices);
        let mut normals = vec![Vec3::ZERO; n];
        frames.gather_normals(&normal_corner_gather(mesh, &indices), &mut normals);
        let mut tangents = vec![Vec4::ZERO; n];
        frames.gather_tangents(&own_corner_gather(n, &indices), &normals, &mut tangents);
        let (lo, hi) = positions
            .iter()
            .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        let extent = if n == 0 { 0.0 } else { (hi - lo).length() };
        Ok(Self {
            positions: positions.to_vec(),
            normals,
            tangents,
            uvs: mesh.uvs().to_vec(),
            indices,
            triangle_tiles,
            triangle_normals: frames.normals,
            vertex_regions,
            position_keys: mesh.position_keys().to_vec(),
            extent,
        })
    }

    /// The vertex positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// The unit vertex normals.
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// The vertex tangents: unit `xyz` along increasing U, `w = ±1` the handedness
    /// (`w · (n × t)` points along increasing V).
    pub fn tangents(&self) -> &[Vec4] {
        &self.tangents
    }

    /// The vertex UVs (tile-local, origin top-left, V down).
    pub fn uvs(&self) -> &[Vec2] {
        &self.uvs
    }

    /// The triangle list, three vertex indices per triangle, counter-clockwise from outside.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The UV tile of every triangle.
    pub fn triangle_tiles(&self) -> &[u16] {
        &self.triangle_tiles
    }

    /// The unit geometric normal of every triangle (zero when degenerate).
    pub fn triangle_normals(&self) -> &[Vec3] {
        &self.triangle_normals
    }

    /// The region of every vertex.
    pub fn vertex_regions(&self) -> &[u16] {
        &self.vertex_regions
    }

    /// The weld key of every vertex (the lowest vertex with the same position).
    pub fn position_keys(&self) -> &[u32] {
        &self.position_keys
    }

    /// The number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.triangle_tiles.len()
    }

    /// The diagonal of the bounding box, metres (scales the ray offsets).
    pub fn extent(&self) -> f32 {
        self.extent
    }

    /// How far a ray starts off the surface, along the triangle normal: `10⁻⁵` of the extent
    /// (20 µm on a 2 m character), at least `10⁻⁷` m.
    pub fn ray_offset(&self) -> f32 {
        (self.extent * 1e-5).max(1e-7)
    }

    /// The hierarchy over this mesh's triangles, for the ray-traced bakers.
    pub fn bvh(&self) -> Result<bvh::Bvh> {
        bvh::Bvh::build(&self.positions, &self.indices)
    }

    /// A per-vertex value (`values[v]` for vertex `v`) interpolated at a surface point; 0 when
    /// `values` is too short.
    pub fn interpolate(&self, p: &SurfacePoint, values: &[f32]) -> f32 {
        let Some(corners) = self.triangle(p.triangle as usize) else { return 0.0 };
        let mut sum = 0.0;
        for (&v, &w) in corners.iter().zip(&p.weights) {
            sum += values.get(v).copied().unwrap_or(0.0) * w;
        }
        sum
    }

    /// The three vertex indices of triangle `t`, if it exists.
    pub fn triangle(&self, t: usize) -> Option<[usize; 3]> {
        let s = self.indices.get(3 * t..3 * t + 3)?;
        Some([s[0] as usize, s[1] as usize, s[2] as usize])
    }

    /// The surface point of triangle `t` at barycentric `(b1, b2)` (`b0 = 1 − b1 − b2`).
    pub fn point(&self, t: usize, b1: f32, b2: f32) -> Option<SurfacePoint> {
        let [a, b, c] = self.triangle(t)?;
        let b0 = 1.0 - b1 - b2;
        let lerp3 = |v: &[Vec3]| v[a] * b0 + v[b] * b1 + v[c] * b2;
        let position = lerp3(&self.positions);
        let geometric = self.triangle_normals[t];
        let normal = lerp3(&self.normals).normalize_or(geometric.normalize_or(Vec3::Z));
        let t4 = self.tangents[a] * b0 + self.tangents[b] * b1 + self.tangents[c] * b2;
        let ortho = t4.truncate() - normal * normal.dot(t4.truncate());
        let tangent = ortho.try_normalize().unwrap_or_else(|| normal.any_orthonormal_vector());
        let sign = if t4.w < 0.0 { -1.0 } else { 1.0 };
        let bitangent = normal.cross(tangent) * sign;
        let corner = if b0 >= b1 && b0 >= b2 {
            a
        } else if b1 >= b2 {
            b
        } else {
            c
        };
        let geometric = if geometric == Vec3::ZERO { normal } else { geometric };
        Some(SurfacePoint {
            position,
            normal,
            geometric,
            tangent,
            bitangent,
            sign,
            region: self.vertex_regions[corner],
            triangle: t as u32,
            weights: [b0, b1, b2],
        })
    }
}

/// One point of the surface with its frame, as the bakers see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// Position, metres.
    pub position: Vec3,
    /// Interpolated unit shading normal.
    pub normal: Vec3,
    /// The unit geometric normal of the triangle (the shading normal on a degenerate one).
    pub geometric: Vec3,
    /// Unit tangent along increasing U, orthogonal to `normal`.
    pub tangent: Vec3,
    /// Unit bitangent `w · (normal × tangent)`, along increasing V.
    pub bitangent: Vec3,
    /// The handedness `w` of the tangent frame (±1).
    pub sign: f32,
    /// The region of the nearest corner.
    pub region: u16,
    /// The triangle the point is on.
    pub triangle: u32,
    /// The barycentric weights of the triangle's three corners.
    pub weights: [f32; 3],
}

impl SurfacePoint {
    /// `v` (object space) in this point's tangent frame: `(v·t, v·b, v·n)`.
    pub fn to_tangent(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.tangent), v.dot(self.bitangent), v.dot(self.normal))
    }
}

/// Which UV tile a baker fills and at what size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BakeTarget {
    /// The UV tile (triangles of other tiles are skipped).
    pub tile: u16,
    /// Width in texels, 1 to [`MAX_MAP_SIZE`].
    pub width: u32,
    /// Height in texels, 1 to [`MAX_MAP_SIZE`].
    pub height: u32,
}

impl BakeTarget {
    /// A `width × height` map of tile `tile`.
    pub const fn new(tile: u16, width: u32, height: u32) -> Self {
        Self { tile, width, height }
    }

    /// Checks the size.
    pub fn check(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > MAX_MAP_SIZE || self.height > MAX_MAP_SIZE {
            return Err(BakeError::invalid("the map size is outside 1..=16384"));
        }
        Ok(())
    }

    /// The number of texels.
    pub const fn texel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// What a texel of a [`BakeMap`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Coverage {
    /// Nothing: the background between UV islands.
    Empty = 0,
    /// The texel centre lies on the surface: a baked value.
    Surface = 1,
    /// A value copied outward by seam padding ([`dilate`]).
    Padding = 2,
}

/// A baked float map: `width × height` texels of `channels` interleaved values, rows top to bottom,
/// and a coverage flag per texel.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeMap {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Values per texel, 1 to [`MAX_CHANNELS`].
    pub channels: u8,
    /// The values, `width · height · channels`.
    pub data: Vec<f32>,
    /// What each texel holds.
    pub coverage: Vec<Coverage>,
}

impl BakeMap {
    /// An empty map filled with `fill`.
    pub fn new(target: &BakeTarget, channels: u8, fill: f32) -> Result<Self> {
        target.check()?;
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(BakeError::invalid("a map has 1 to 4 channels"));
        }
        let n = target.texel_count();
        Ok(Self {
            width: target.width,
            height: target.height,
            channels,
            data: vec![fill; n * channels as usize],
            coverage: vec![Coverage::Empty; n],
        })
    }

    /// The number of texels.
    pub const fn texel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// The values of texel `i` (row-major), if it exists.
    pub fn texel(&self, i: usize) -> Option<&[f32]> {
        let c = self.channels as usize;
        self.data.get(i * c..i * c + c)
    }

    /// The values of texel `(x, y)`, if it exists.
    pub fn at(&self, x: u32, y: u32) -> Option<&[f32]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.texel(y as usize * self.width as usize + x as usize)
    }

    /// Sets texel `i` to `values` (as many as the map has channels) and marks it covered.
    pub fn set(&mut self, i: usize, values: &[f32], coverage: Coverage) {
        let c = self.channels as usize;
        if let Some(slot) = self.data.get_mut(i * c..i * c + c) {
            for (d, &v) in slot.iter_mut().zip(values) {
                *d = v;
            }
            if let Some(cov) = self.coverage.get_mut(i) {
                *cov = coverage;
            }
        }
    }

    /// The number of texels with each coverage: `(empty, surface, padding)`.
    pub fn coverage_counts(&self) -> (usize, usize, usize) {
        self.coverage.iter().fold((0, 0, 0), |(e, s, p), c| match c {
            Coverage::Empty => (e + 1, s, p),
            Coverage::Surface => (e, s + 1, p),
            Coverage::Padding => (e, s, p + 1),
        })
    }

    /// The map as a 32-bit float image with colour space `space`.
    pub fn to_image(&self, space: ColorSpace) -> Result<Image> {
        Ok(Image::new(Format::Ktx2, self.width, self.height, self.channels, space, Samples::F32(self.data.clone()))?)
    }

    /// The map as an 8-bit image: every value becomes `round(255 · clamp(v · scale + bias, 0, 1))`.
    /// For data maps (`space` is `Data` or `Normal`) and for looking at a map.
    pub fn to_unorm8(&self, space: ColorSpace, scale: f32, bias: f32) -> Result<Image> {
        let samples = self.data.iter().map(|&v| ((v * scale + bias).clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        Ok(Image::new(Format::Ktx2, self.width, self.height, self.channels, space, Samples::U8(samples))?)
    }

    /// The map of one channel `c` of this map.
    pub fn channel(&self, c: u8) -> Result<Self> {
        if c >= self.channels {
            return Err(BakeError::invalid("the map has no such channel"));
        }
        let data = self.data.chunks_exact(self.channels as usize).map(|t| t[c as usize]).collect();
        Ok(Self { width: self.width, height: self.height, channels: 1, data, coverage: self.coverage.clone() })
    }
}

/// Synthetic test shapes and the bakers' cross-module tests (regression images).
#[cfg(test)]
pub(crate) mod tests;
