//! The glTF 2.0 document model: every top-level array of the specification, typed, with indices as
//! `usize`, plus the ratified extensions the studio reads and writes `[IO-10]`.
//!
//! Every object keeps the `extensions` this crate does not interpret and its `extras` in an
//! [`Extra`], so a document read and written again loses nothing it understood and keeps the rest.
//! Typed extensions: `KHR_lights_punctual`, `KHR_texture_transform`, `KHR_texture_basisu`,
//! `KHR_materials_emissive_strength`, `_ior`, `_specular`, `_clearcoat`, `_sheen`, `_transmission`,
//! `_volume`, `_diffuse_transmission`, `_unlit`; `KHR_mesh_quantization` widens the accepted
//! attribute component types.

use crate::json::{Object, Value};

/// The extensions this reader implements; a file that requires any other is refused.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "KHR_lights_punctual",
    "KHR_materials_clearcoat",
    "KHR_materials_diffuse_transmission",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_unlit",
    "KHR_materials_volume",
    "KHR_mesh_quantization",
    "KHR_texture_basisu",
    "KHR_texture_transform",
];

/// The uninterpreted `extensions` and the `extras` of an object.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extra {
    /// Extension objects this crate does not interpret, by extension name, in file order.
    pub extensions: Object,
    /// The `extras` value, if present (studio payloads: rig logic, region ids, landmarks).
    pub extras: Option<Value>,
}

impl Extra {
    /// Whether there is nothing to write.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty() && self.extras.is_none()
    }
}

/// The whole document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Document {
    /// `asset`.
    pub asset: Asset,
    /// `extensionsUsed`.
    pub extensions_used: Vec<String>,
    /// `extensionsRequired`.
    pub extensions_required: Vec<String>,
    /// `scene`: the default scene.
    pub scene: Option<usize>,
    /// `scenes`.
    pub scenes: Vec<Scene>,
    /// `nodes`.
    pub nodes: Vec<Node>,
    /// `meshes`.
    pub meshes: Vec<Mesh>,
    /// `skins`.
    pub skins: Vec<Skin>,
    /// `accessors`.
    pub accessors: Vec<Accessor>,
    /// `bufferViews`.
    pub buffer_views: Vec<BufferView>,
    /// `buffers`.
    pub buffers: Vec<Buffer>,
    /// `materials`.
    pub materials: Vec<Material>,
    /// `textures`.
    pub textures: Vec<Texture>,
    /// `images`.
    pub images: Vec<Image>,
    /// `samplers`.
    pub samplers: Vec<Sampler>,
    /// `animations`.
    pub animations: Vec<Animation>,
    /// `cameras`.
    pub cameras: Vec<Camera>,
    /// `extensions.KHR_lights_punctual.lights`.
    pub lights: Vec<Light>,
    /// Root extensions and extras.
    pub extra: Extra,
}

/// `asset`: version and provenance of the file.
#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    /// `version`, `2.0` for files this crate writes; any `2.x` is read.
    pub version: String,
    /// `minVersion`.
    pub min_version: Option<String>,
    /// `generator`.
    pub generator: Option<String>,
    /// `copyright`.
    pub copyright: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

impl Default for Asset {
    fn default() -> Self {
        Self { version: "2.0".to_owned(), min_version: None, generator: None, copyright: None, extra: Extra::default() }
    }
}

/// A scene: a set of root nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    /// `name`.
    pub name: Option<String>,
    /// `nodes`: the root nodes.
    pub nodes: Vec<usize>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// A node's local transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    /// `matrix`: column-major 4×4.
    Matrix([f32; 16]),
    /// `translation`, `rotation` (unit quaternion `x, y, z, w`) and `scale`.
    Trs {
        /// Translation.
        translation: [f32; 3],
        /// Rotation quaternion `x, y, z, w`.
        rotation: [f32; 4],
        /// Scale.
        scale: [f32; 3],
    },
}

impl Transform {
    /// The identity transform, written as no members at all.
    pub const IDENTITY: Self = Self::Trs { translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] };

    /// The transform as a column-major matrix (`M = T * R * S`).
    #[must_use]
    pub fn matrix(&self) -> [f32; 16] {
        match *self {
            Self::Matrix(m) => m,
            Self::Trs { translation: t, rotation: [x, y, z, w], scale: s } => {
                let (xx, yy, zz) = (x * x, y * y, z * z);
                let (xy, xz, yz, wx, wy, wz) = (x * y, x * z, y * z, w * x, w * y, w * z);
                [
                    (1.0 - 2.0 * (yy + zz)) * s[0],
                    2.0 * (xy + wz) * s[0],
                    2.0 * (xz - wy) * s[0],
                    0.0,
                    2.0 * (xy - wz) * s[1],
                    (1.0 - 2.0 * (xx + zz)) * s[1],
                    2.0 * (yz + wx) * s[1],
                    0.0,
                    2.0 * (xz + wy) * s[2],
                    2.0 * (yz - wx) * s[2],
                    (1.0 - 2.0 * (xx + yy)) * s[2],
                    0.0,
                    t[0],
                    t[1],
                    t[2],
                    1.0,
                ]
            }
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A node of the scene hierarchy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    /// `name`.
    pub name: Option<String>,
    /// `children`.
    pub children: Vec<usize>,
    /// `camera`.
    pub camera: Option<usize>,
    /// `mesh`.
    pub mesh: Option<usize>,
    /// `skin`.
    pub skin: Option<usize>,
    /// `matrix` or `translation` / `rotation` / `scale`.
    pub transform: Transform,
    /// `weights`: morph target weights overriding the mesh's.
    pub weights: Vec<f32>,
    /// `extensions.KHR_lights_punctual.light`.
    pub light: Option<usize>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// A mesh: primitives sharing morph target weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// `name`.
    pub name: Option<String>,
    /// `primitives`.
    pub primitives: Vec<Primitive>,
    /// `weights`: default morph target weights.
    pub weights: Vec<f32>,
    /// Extensions and extras (`extras.targetNames` names the morph targets by convention).
    pub extra: Extra,
}

impl Mesh {
    /// The morph target names from `extras.targetNames` (the common convention), empty when absent
    /// or malformed.
    #[must_use]
    pub fn target_names(&self) -> Vec<&str> {
        let names = self.extra.extras.as_ref().and_then(|e| e.get("targetNames")).and_then(Value::as_array);
        names.map(|n| n.iter().filter_map(Value::as_str).collect()).unwrap_or_default()
    }

    /// Sets `extras.targetNames`, keeping the other members of an `extras` object (an `extras` that
    /// is not an object is replaced).
    pub fn set_target_names(&mut self, names: &[&str]) {
        let list = Value::Array(names.iter().map(|&n| Value::from(n)).collect());
        match &mut self.extra.extras {
            Some(Value::Object(o)) => {
                o.insert("targetNames", list);
            }
            slot => {
                let mut o = Object::new();
                o.insert("targetNames", list);
                *slot = Some(Value::Object(o));
            }
        }
    }
}

/// The topology of a primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// 0.
    Points,
    /// 1.
    Lines,
    /// 2.
    LineLoop,
    /// 3.
    LineStrip,
    /// 4, the default.
    #[default]
    Triangles,
    /// 5.
    TriangleStrip,
    /// 6.
    TriangleFan,
}

impl Mode {
    /// The `mode` value.
    #[must_use]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// The mode of a `mode` value.
    #[must_use]
    pub const fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            0 => Self::Points,
            1 => Self::Lines,
            2 => Self::LineLoop,
            3 => Self::LineStrip,
            4 => Self::Triangles,
            5 => Self::TriangleStrip,
            6 => Self::TriangleFan,
            _ => return None,
        })
    }
}

/// A named attribute of a primitive or morph target and its accessor.
pub type Attribute = (String, usize);

/// A primitive: geometry to draw with one material.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Primitive {
    /// `attributes`, in file order.
    pub attributes: Vec<Attribute>,
    /// `indices`.
    pub indices: Option<usize>,
    /// `material`.
    pub material: Option<usize>,
    /// `mode`.
    pub mode: Mode,
    /// `targets`: morph targets, each a set of attribute deltas.
    pub targets: Vec<Vec<Attribute>>,
    /// Extensions and extras.
    pub extra: Extra,
}

impl Primitive {
    /// The accessor of attribute `name`.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<usize> {
        find(&self.attributes, name)
    }
}

/// The accessor of attribute `name` in a list of attributes.
#[must_use]
pub fn find(attributes: &[Attribute], name: &str) -> Option<usize> {
    attributes.iter().find(|(n, _)| n == name).map(|&(_, a)| a)
}

/// The component type of an accessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// 5120, `i8`.
    I8,
    /// 5121, `u8`.
    U8,
    /// 5122, `i16`.
    I16,
    /// 5123, `u16`.
    U16,
    /// 5125, `u32`.
    U32,
    /// 5126, `f32`.
    F32,
}

impl ComponentType {
    /// The `componentType` value.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::I8 => 5120,
            Self::U8 => 5121,
            Self::I16 => 5122,
            Self::U16 => 5123,
            Self::U32 => 5125,
            Self::F32 => 5126,
        }
    }

    /// The component type of a `componentType` value.
    #[must_use]
    pub const fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            5120 => Self::I8,
            5121 => Self::U8,
            5122 => Self::I16,
            5123 => Self::U16,
            5125 => Self::U32,
            5126 => Self::F32,
            _ => return None,
        })
    }

    /// Bytes per component.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::U32 | Self::F32 => 4,
        }
    }

    /// Whether the type is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::F32)
    }
}

/// The element type of an accessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// `SCALAR`.
    Scalar,
    /// `VEC2`.
    Vec2,
    /// `VEC3`.
    Vec3,
    /// `VEC4`.
    Vec4,
    /// `MAT2`.
    Mat2,
    /// `MAT3`.
    Mat3,
    /// `MAT4`.
    Mat4,
}

impl ElementType {
    /// The names, as written in a file.
    pub const NAMES: [(&'static str, Self); 7] = [
        ("SCALAR", Self::Scalar),
        ("VEC2", Self::Vec2),
        ("VEC3", Self::Vec3),
        ("VEC4", Self::Vec4),
        ("MAT2", Self::Mat2),
        ("MAT3", Self::Mat3),
        ("MAT4", Self::Mat4),
    ];

    /// The name written in a file.
    #[must_use]
    pub const fn name(self) -> &'static str {
        Self::NAMES[self as usize].0
    }

    /// Components per element.
    #[must_use]
    pub const fn components(self) -> usize {
        let (cols, rows) = self.shape();
        cols * rows
    }

    /// Columns and rows (1 column for scalars and vectors).
    #[must_use]
    pub const fn shape(self) -> (usize, usize) {
        match self {
            Self::Scalar => (1, 1),
            Self::Vec2 => (1, 2),
            Self::Vec3 => (1, 3),
            Self::Vec4 => (1, 4),
            Self::Mat2 => (2, 2),
            Self::Mat3 => (3, 3),
            Self::Mat4 => (4, 4),
        }
    }

    /// The byte distance between matrix columns: each column starts on a 4-byte boundary, as the
    /// specification requires for matrices of 1- and 2-byte components.
    #[must_use]
    pub const fn column_stride(self, component: ComponentType) -> usize {
        let (cols, rows) = self.shape();
        let raw = rows * component.size();
        if cols > 1 { raw.next_multiple_of(4) } else { raw }
    }

    /// The byte size of one element.
    #[must_use]
    pub const fn size(self, component: ComponentType) -> usize {
        self.shape().0 * self.column_stride(component)
    }
}

/// A typed view of buffer data.
#[derive(Clone, Debug, PartialEq)]
pub struct Accessor {
    /// `bufferView`; when absent the data is zeros (before sparse substitution).
    pub buffer_view: Option<usize>,
    /// `byteOffset` within the buffer view.
    pub byte_offset: usize,
    /// `componentType`.
    pub component_type: ComponentType,
    /// `normalized`.
    pub normalized: bool,
    /// `count`: number of elements.
    pub count: usize,
    /// `type`.
    pub kind: ElementType,
    /// `min`, one value per component (empty when absent).
    pub min: Vec<f64>,
    /// `max`, one value per component (empty when absent).
    pub max: Vec<f64>,
    /// `sparse`.
    pub sparse: Option<Sparse>,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// Sparse substitution of some elements of an accessor.
#[derive(Clone, Debug, PartialEq)]
pub struct Sparse {
    /// `count`: number of substituted elements.
    pub count: usize,
    /// `indices.bufferView`.
    pub indices_view: usize,
    /// `indices.byteOffset`.
    pub indices_offset: usize,
    /// `indices.componentType` (`U8`, `U16` or `U32`).
    pub indices_type: ComponentType,
    /// `values.bufferView`.
    pub values_view: usize,
    /// `values.byteOffset`.
    pub values_offset: usize,
    /// Extensions and extras of the `sparse` object.
    pub extra: Extra,
}

/// The intended GPU binding of a buffer view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// 34962, vertex attributes.
    ArrayBuffer,
    /// 34963, indices.
    ElementArrayBuffer,
}

impl Target {
    /// The `target` value.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::ArrayBuffer => 34962,
            Self::ElementArrayBuffer => 34963,
        }
    }
}

/// A range of a buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferView {
    /// `buffer`.
    pub buffer: usize,
    /// `byteOffset`.
    pub byte_offset: usize,
    /// `byteLength`.
    pub byte_length: usize,
    /// `byteStride`: the distance between vertex attribute elements (4 to 252, a multiple of 4).
    pub byte_stride: Option<usize>,
    /// `target`.
    pub target: Option<Target>,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// A block of binary data.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    /// `uri`: a relative reference or a `data:` URI; absent for the GLB BIN chunk.
    pub uri: Option<String>,
    /// `byteLength`.
    pub byte_length: usize,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// An image, by URI or in a buffer view. The pixels are not decoded here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Image {
    /// `uri`.
    pub uri: Option<String>,
    /// `mimeType` (`image/png`, `image/jpeg`, `image/ktx2`, ...).
    pub mime_type: Option<String>,
    /// `bufferView`.
    pub buffer_view: Option<usize>,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// Texture filtering and wrapping, as the WebGL enum values of the specification.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampler {
    /// `magFilter`: 9728 nearest, 9729 linear.
    pub mag_filter: Option<u32>,
    /// `minFilter`: 9728, 9729, or 9984 to 9987 (mipmapped).
    pub min_filter: Option<u32>,
    /// `wrapS`: 33071 clamp, 33648 mirrored repeat, 10497 repeat (the default).
    pub wrap_s: u32,
    /// `wrapT`.
    pub wrap_t: u32,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// The `wrapS` / `wrapT` default: repeat.
pub const REPEAT: u32 = 10497;

impl Default for Sampler {
    fn default() -> Self {
        Self { mag_filter: None, min_filter: None, wrap_s: REPEAT, wrap_t: REPEAT, name: None, extra: Extra::default() }
    }
}

/// A texture: an image and a sampler.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Texture {
    /// `sampler`.
    pub sampler: Option<usize>,
    /// `source`: a PNG or JPEG image.
    pub source: Option<usize>,
    /// `extensions.KHR_texture_basisu.source`: a KTX2 image.
    pub basisu_source: Option<usize>,
    /// `name`.
    pub name: Option<String>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// `KHR_texture_transform`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    /// `offset`.
    pub offset: [f32; 2],
    /// `rotation` in radians.
    pub rotation: f32,
    /// `scale`.
    pub scale: [f32; 2],
    /// `texCoord`: overrides the texture info's.
    pub tex_coord: Option<u32>,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self { offset: [0.0; 2], rotation: 0.0, scale: [1.0; 2], tex_coord: None }
    }
}

/// A reference from a material to a texture.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureInfo {
    /// `index`.
    pub index: usize,
    /// `texCoord`: which `TEXCOORD_n` set.
    pub tex_coord: u32,
    /// `scale` of a normal texture or `strength` of an occlusion texture (1 when absent); unused
    /// elsewhere.
    pub scale: f32,
    /// `extensions.KHR_texture_transform`.
    pub transform: Option<TextureTransform>,
    /// Extensions and extras.
    pub extra: Extra,
}

impl TextureInfo {
    /// A reference to texture `index` with the defaults.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self { index, tex_coord: 0, scale: 1.0, transform: None, extra: Extra::default() }
    }
}

/// How alpha is used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// `OPAQUE`, the default.
    #[default]
    Opaque,
    /// `MASK` with the material's cutoff.
    Mask,
    /// `BLEND`.
    Blend,
}

/// `pbrMetallicRoughness`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pbr {
    /// `baseColorFactor`, linear RGBA.
    pub base_color_factor: [f32; 4],
    /// `baseColorTexture` (sRGB).
    pub base_color_texture: Option<TextureInfo>,
    /// `metallicFactor`.
    pub metallic_factor: f32,
    /// `roughnessFactor`.
    pub roughness_factor: f32,
    /// `metallicRoughnessTexture` (roughness in G, metalness in B).
    pub metallic_roughness_texture: Option<TextureInfo>,
    /// Extensions and extras.
    pub extra: Extra,
}

impl Default for Pbr {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0; 4],
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            extra: Extra::default(),
        }
    }
}

/// `KHR_materials_specular`.
#[derive(Clone, Debug, PartialEq)]
pub struct Specular {
    /// `specularFactor`.
    pub factor: f32,
    /// `specularTexture` (A channel).
    pub texture: Option<TextureInfo>,
    /// `specularColorFactor`.
    pub color_factor: [f32; 3],
    /// `specularColorTexture` (RGB, sRGB).
    pub color_texture: Option<TextureInfo>,
}

impl Default for Specular {
    fn default() -> Self {
        Self { factor: 1.0, texture: None, color_factor: [1.0; 3], color_texture: None }
    }
}

/// `KHR_materials_clearcoat` (the eye's wet cornea layer, for instance).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clearcoat {
    /// `clearcoatFactor`.
    pub factor: f32,
    /// `clearcoatTexture` (R).
    pub texture: Option<TextureInfo>,
    /// `clearcoatRoughnessFactor`.
    pub roughness_factor: f32,
    /// `clearcoatRoughnessTexture` (G).
    pub roughness_texture: Option<TextureInfo>,
    /// `clearcoatNormalTexture`.
    pub normal_texture: Option<TextureInfo>,
}

/// `KHR_materials_sheen` (vellus hair, for instance).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sheen {
    /// `sheenColorFactor`.
    pub color_factor: [f32; 3],
    /// `sheenColorTexture` (RGB, sRGB).
    pub color_texture: Option<TextureInfo>,
    /// `sheenRoughnessFactor`.
    pub roughness_factor: f32,
    /// `sheenRoughnessTexture` (A).
    pub roughness_texture: Option<TextureInfo>,
}

/// `KHR_materials_transmission`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transmission {
    /// `transmissionFactor`.
    pub factor: f32,
    /// `transmissionTexture` (R).
    pub texture: Option<TextureInfo>,
}

/// `KHR_materials_volume`.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume {
    /// `thicknessFactor`.
    pub thickness_factor: f32,
    /// `thicknessTexture` (G).
    pub thickness_texture: Option<TextureInfo>,
    /// `attenuationDistance`; `None` is infinite.
    pub attenuation_distance: Option<f32>,
    /// `attenuationColor`.
    pub attenuation_color: [f32; 3],
}

impl Default for Volume {
    fn default() -> Self {
        Self { thickness_factor: 0.0, thickness_texture: None, attenuation_distance: None, attenuation_color: [1.0; 3] }
    }
}

/// `KHR_materials_diffuse_transmission` (light through thin skin such as ears).
#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseTransmission {
    /// `diffuseTransmissionFactor`.
    pub factor: f32,
    /// `diffuseTransmissionTexture` (A).
    pub texture: Option<TextureInfo>,
    /// `diffuseTransmissionColorFactor`.
    pub color_factor: [f32; 3],
    /// `diffuseTransmissionColorTexture` (RGB, sRGB).
    pub color_texture: Option<TextureInfo>,
}

impl Default for DiffuseTransmission {
    fn default() -> Self {
        Self { factor: 0.0, texture: None, color_factor: [1.0; 3], color_texture: None }
    }
}

/// A material: metallic-roughness PBR plus the typed `KHR_materials_*` extensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    /// `name`.
    pub name: Option<String>,
    /// `pbrMetallicRoughness`.
    pub pbr: Pbr,
    /// `normalTexture` (its `scale` in [`TextureInfo::scale`]).
    pub normal_texture: Option<TextureInfo>,
    /// `occlusionTexture` (its `strength` in [`TextureInfo::scale`]).
    pub occlusion_texture: Option<TextureInfo>,
    /// `emissiveTexture`.
    pub emissive_texture: Option<TextureInfo>,
    /// `emissiveFactor`.
    pub emissive_factor: [f32; 3],
    /// `alphaMode`.
    pub alpha_mode: AlphaMode,
    /// `alphaCutoff` (0.5 when absent).
    pub alpha_cutoff: f32,
    /// `doubleSided`.
    pub double_sided: bool,
    /// `KHR_materials_unlit`.
    pub unlit: bool,
    /// `KHR_materials_emissive_strength.emissiveStrength`.
    pub emissive_strength: Option<f32>,
    /// `KHR_materials_ior.ior`.
    pub ior: Option<f32>,
    /// `KHR_materials_specular`.
    pub specular: Option<Specular>,
    /// `KHR_materials_clearcoat`.
    pub clearcoat: Option<Clearcoat>,
    /// `KHR_materials_sheen`.
    pub sheen: Option<Sheen>,
    /// `KHR_materials_transmission`.
    pub transmission: Option<Transmission>,
    /// `KHR_materials_volume`.
    pub volume: Option<Volume>,
    /// `KHR_materials_diffuse_transmission`.
    pub diffuse_transmission: Option<DiffuseTransmission>,
    /// Extensions and extras (remaining skin and eye parameters go to `extras`).
    pub extra: Extra,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: None,
            pbr: Pbr::default(),
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            emissive_factor: [0.0; 3],
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
            emissive_strength: None,
            ior: None,
            specular: None,
            clearcoat: None,
            sheen: None,
            transmission: None,
            volume: None,
            diffuse_transmission: None,
            extra: Extra::default(),
        }
    }
}

impl Material {
    /// Every texture reference of the material, core and extensions.
    #[must_use]
    pub fn texture_infos<'a>(&'a self) -> Vec<&'a TextureInfo> {
        let mut out: Vec<&TextureInfo> = Vec::new();
        let mut add = |t: &'a Option<TextureInfo>| {
            if let Some(t) = t {
                out.push(t);
            }
        };
        add(&self.pbr.base_color_texture);
        add(&self.pbr.metallic_roughness_texture);
        add(&self.normal_texture);
        add(&self.occlusion_texture);
        add(&self.emissive_texture);
        if let Some(s) = &self.specular {
            add(&s.texture);
            add(&s.color_texture);
        }
        if let Some(c) = &self.clearcoat {
            add(&c.texture);
            add(&c.roughness_texture);
            add(&c.normal_texture);
        }
        if let Some(s) = &self.sheen {
            add(&s.color_texture);
            add(&s.roughness_texture);
        }
        if let Some(t) = &self.transmission {
            add(&t.texture);
        }
        if let Some(v) = &self.volume {
            add(&v.thickness_texture);
        }
        if let Some(d) = &self.diffuse_transmission {
            add(&d.texture);
            add(&d.color_texture);
        }
        out
    }
}

/// A skin: joints and their inverse bind matrices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skin {
    /// `name`.
    pub name: Option<String>,
    /// `inverseBindMatrices`: a `MAT4` float accessor with one matrix per joint.
    pub inverse_bind_matrices: Option<usize>,
    /// `skeleton`: the common root.
    pub skeleton: Option<usize>,
    /// `joints`: node indices.
    pub joints: Vec<usize>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// A camera projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// `perspective`.
    Perspective {
        /// `aspectRatio`.
        aspect_ratio: Option<f32>,
        /// `yfov` in radians.
        yfov: f32,
        /// `zfar`; `None` is an infinite projection.
        zfar: Option<f32>,
        /// `znear`.
        znear: f32,
    },
    /// `orthographic`.
    Orthographic {
        /// `xmag`.
        xmag: f32,
        /// `ymag`.
        ymag: f32,
        /// `zfar`.
        zfar: f32,
        /// `znear`.
        znear: f32,
    },
}

/// A camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// `name`.
    pub name: Option<String>,
    /// `type` and its projection object.
    pub projection: Projection,
    /// Extensions and extras.
    pub extra: Extra,
}

/// The kind of a punctual light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    /// `directional`.
    Directional,
    /// `point`.
    Point,
    /// `spot` with its cone angles in radians.
    Spot {
        /// `innerConeAngle` (0 when absent).
        inner_cone_angle: f32,
        /// `outerConeAngle` (π/4 when absent).
        outer_cone_angle: f32,
    },
}

/// A `KHR_lights_punctual` light.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    /// `name`.
    pub name: Option<String>,
    /// `color`, linear.
    pub color: [f32; 3],
    /// `intensity` (candela for point and spot, lux for directional).
    pub intensity: f32,
    /// `type` and the spot cone.
    pub kind: LightKind,
    /// `range`; `None` is infinite.
    pub range: Option<f32>,
    /// Extensions and extras.
    pub extra: Extra,
}

/// The property an animation channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Path {
    /// `translation`.
    Translation,
    /// `rotation`.
    Rotation,
    /// `scale`.
    Scale,
    /// `weights`: morph target weights.
    Weights,
    /// `pointer` (`KHR_animation_pointer`, kept in the target's extensions).
    Pointer,
}

impl Path {
    /// The names, as written in a file.
    pub const NAMES: [(&'static str, Self); 5] = [
        ("translation", Self::Translation),
        ("rotation", Self::Rotation),
        ("scale", Self::Scale),
        ("weights", Self::Weights),
        ("pointer", Self::Pointer),
    ];

    /// The name written in a file.
    #[must_use]
    pub const fn name(self) -> &'static str {
        Self::NAMES[self as usize].0
    }
}

/// How keyframes are interpolated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// `LINEAR` (spherical for rotations), the default.
    #[default]
    Linear,
    /// `STEP`.
    Step,
    /// `CUBICSPLINE`: each output is an in-tangent, a value and an out-tangent.
    CubicSpline,
}

/// An animation channel: a sampler driving a node property.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// `sampler`: index into the animation's samplers.
    pub sampler: usize,
    /// `target.node`.
    pub node: Option<usize>,
    /// `target.path`.
    pub path: Path,
    /// Extensions and extras of the channel.
    pub extra: Extra,
    /// Extensions and extras of the target (`KHR_animation_pointer` lives here).
    pub target_extra: Extra,
}

/// An animation sampler: keyframe times and values.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimationSampler {
    /// `input`: keyframe times, a `SCALAR` float accessor.
    pub input: usize,
    /// `output`: keyframe values.
    pub output: usize,
    /// `interpolation`.
    pub interpolation: Interpolation,
    /// Extensions and extras.
    pub extra: Extra,
}

/// An animation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Animation {
    /// `name`.
    pub name: Option<String>,
    /// `channels`.
    pub channels: Vec<Channel>,
    /// `samplers`.
    pub samplers: Vec<AnimationSampler>,
    /// Extensions and extras.
    pub extra: Extra,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_sizes_follow_the_column_alignment_rule() {
        use ComponentType::{F32, I8, U8, U16};
        assert_eq!(ElementType::Vec3.size(U8), 3);
        assert_eq!(ElementType::Vec3.size(F32), 12);
        assert_eq!(ElementType::Mat2.size(U8), 8);
        assert_eq!(ElementType::Mat3.size(I8), 12);
        assert_eq!(ElementType::Mat3.size(U16), 24);
        assert_eq!(ElementType::Mat2.size(U16), 8);
        assert_eq!(ElementType::Mat4.size(F32), 64);
        for (name, kind) in ElementType::NAMES {
            assert_eq!(kind.name(), name);
        }
        for (name, path) in Path::NAMES {
            assert_eq!(path.name(), name);
        }
        for code in [5120, 5121, 5122, 5123, 5125, 5126] {
            assert_eq!(ComponentType::from_code(code).map(|c| u64::from(c.code())), Some(code));
        }
        assert_eq!(ComponentType::from_code(5124), None);
        assert_eq!(Mode::from_code(7), None);
        assert_eq!(Mode::from_code(5).map(Mode::code), Some(5));
    }

    #[test]
    fn trs_matrix_matches_hand_computed_values() {
        // A quarter turn about +Y, then translation: +X maps to -Z.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let t = Transform::Trs { translation: [1.0, 2.0, 3.0], rotation: [0.0, h, 0.0, h], scale: [2.0, 1.0, 1.0] };
        let m = t.matrix();
        let x = [m[0], m[1], m[2]];
        assert!((x[0]).abs() < 1e-6 && (x[2] + 2.0).abs() < 1e-6, "{x:?}");
        assert_eq!([m[12], m[13], m[14], m[15]], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            Transform::IDENTITY.matrix(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}
