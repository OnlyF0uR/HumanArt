//! glTF JSON to [`Document`]: member types, enum values and the typed extensions `[IO-10]`.
//!
//! This stage checks only what one object says about itself (kinds, enum values, finite numbers,
//! fixed array lengths). Cross-references (indices, buffer ranges, accessor types, the node
//! hierarchy) are checked by [`super::validate`], which also runs on documents built in code.
//!
//! Errors from this stage are [`Error::Invalid`] with offset 0: the JSON tree keeps no byte
//! positions, so the `what` names the member instead.

use super::doc::{Accessor, Attribute};
use super::doc::{
    AlphaMode, Animation, AnimationSampler, Asset, Buffer, BufferView, Camera, Channel, Clearcoat, ComponentType,
    DiffuseTransmission, Document, ElementType, Extra, Image, Interpolation, Light, LightKind, Material, Mesh, Mode,
    Node, Path, Pbr, Primitive, Projection, Sampler, Scene, Sheen, Skin, Sparse, Specular, Target, Texture,
    TextureInfo, TextureTransform, Transform, Transmission, Volume,
};
use crate::error::{Error, Format, Result};
use crate::json::{Object, Value};

/// An [`Error::Invalid`] about the member `what`.
pub(crate) const fn invalid(what: &'static str) -> Error {
    Error::Invalid { format: Format::Gltf, offset: 0, what }
}

/// An empty object, the stand-in for a missing optional object.
static EMPTY: Object = Object::new();

/// The value as an object.
fn object<'a>(value: &'a Value, what: &'static str) -> Result<&'a Object> {
    value.as_object().ok_or(invalid(what))
}

/// The optional member `key` as an object.
fn opt_object<'a>(o: &'a Object, key: &str, what: &'static str) -> Result<Option<&'a Object>> {
    o.get(key).map(|v| object(v, what)).transpose()
}

/// The optional member `key` as an array; a missing member is an empty array.
fn array<'a>(o: &'a Object, key: &str, what: &'static str) -> Result<&'a [Value]> {
    o.get(key).map_or(Ok(&[][..]), |v| v.as_array().ok_or(invalid(what)))
}

/// The optional member `key` as a string.
fn opt_string(o: &Object, key: &str, what: &'static str) -> Result<Option<String>> {
    o.get(key).map(|v| v.as_str().map(str::to_owned).ok_or(invalid(what))).transpose()
}

/// A whole, non-negative number as `u64`.
fn uint(v: &Value, what: &'static str) -> Result<u64> {
    v.as_u64().ok_or(invalid(what))
}

/// The optional member `key` as a whole, non-negative number.
fn opt_uint(o: &Object, key: &str, what: &'static str) -> Result<Option<u64>> {
    o.get(key).map(|v| uint(v, what)).transpose()
}

/// A whole, non-negative number as an index or a size.
fn usize_of(v: &Value, what: &'static str) -> Result<usize> {
    usize::try_from(uint(v, what)?).map_err(|_| invalid(what))
}

/// The optional member `key` as an index or a size.
fn opt_usize(o: &Object, key: &str, what: &'static str) -> Result<Option<usize>> {
    o.get(key).map(|v| usize_of(v, what)).transpose()
}

/// The required member `key` as an index or a size.
fn req_usize(o: &Object, key: &str, what: &'static str) -> Result<usize> {
    usize_of(o.get(key).ok_or(invalid(what))?, what)
}

/// The optional member `key` as a `u32`.
fn opt_u32(o: &Object, key: &str, what: &'static str) -> Result<Option<u32>> {
    opt_uint(o, key, what)?.map(|v| u32::try_from(v).map_err(|_| invalid(what))).transpose()
}

/// A finite number that is also finite as `f32`.
fn f32_of(v: &Value, what: &'static str) -> Result<f32> {
    let f = v.as_f64().ok_or(invalid(what))?;
    #[allow(clippy::cast_possible_truncation)]
    let narrow = f as f32;
    if narrow.is_finite() { Ok(narrow) } else { Err(invalid(what)) }
}

/// The optional member `key` as an `f32`.
fn opt_f32(o: &Object, key: &str, what: &'static str) -> Result<Option<f32>> {
    o.get(key).map(|v| f32_of(v, what)).transpose()
}

/// The optional member `key` as an `f32`, `default` when absent.
fn f32_or(o: &Object, key: &str, default: f32, what: &'static str) -> Result<f32> {
    Ok(opt_f32(o, key, what)?.unwrap_or(default))
}

/// The optional member `key` as an array of exactly `N` numbers.
fn opt_floats<const N: usize>(o: &Object, key: &str, what: &'static str) -> Result<Option<[f32; N]>> {
    let Some(v) = o.get(key) else { return Ok(None) };
    let items = v.as_array().ok_or(invalid(what))?;
    if items.len() != N {
        return Err(invalid(what));
    }
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = f32_of(item, what)?;
    }
    Ok(Some(out))
}

/// The optional member `key` as `N` numbers, `default` when absent.
fn floats_or<const N: usize>(o: &Object, key: &str, default: [f32; N], what: &'static str) -> Result<[f32; N]> {
    Ok(opt_floats(o, key, what)?.unwrap_or(default))
}

/// The optional member `key` as a boolean.
fn bool_or(o: &Object, key: &str, default: bool, what: &'static str) -> Result<bool> {
    o.get(key).map_or(Ok(default), |v| v.as_bool().ok_or(invalid(what)))
}

/// The optional member `key` as an array of indices.
fn indices(o: &Object, key: &str, what: &'static str) -> Result<Vec<usize>> {
    array(o, key, what)?.iter().map(|v| usize_of(v, what)).collect()
}

/// The optional member `key` as an array of floats.
fn float_list(o: &Object, key: &str, what: &'static str) -> Result<Vec<f32>> {
    array(o, key, what)?.iter().map(|v| f32_of(v, what)).collect()
}

/// The optional member `key` as an array of strings.
fn strings(o: &Object, key: &str, what: &'static str) -> Result<Vec<String>> {
    array(o, key, what)?.iter().map(|v| v.as_str().map(str::to_owned).ok_or(invalid(what))).collect()
}

/// The `extensions` and `extras` of `o`, minus the typed extensions named in `typed`.
fn extra(o: &Object, typed: &[&str]) -> Result<Extra> {
    let mut extensions = Object::new();
    if let Some(ext) = opt_object(o, "extensions", "extensions is not an object")? {
        for (name, value) in ext {
            if !typed.contains(&name) {
                extensions.insert(name, value.clone());
            }
        }
    }
    Ok(Extra { extensions, extras: o.get("extras").cloned() })
}

/// The extension object `name` of `o`, if present.
fn extension<'a>(o: &'a Object, name: &str) -> Result<Option<&'a Object>> {
    match opt_object(o, "extensions", "extensions is not an object")? {
        Some(ext) => opt_object(ext, name, "extension is not an object"),
        None => Ok(None),
    }
}

/// The objects of the top-level array `key`.
fn each<'a, T>(root: &'a Object, key: &str, what: &'static str, f: impl Fn(&'a Object) -> Result<T>) -> Result<Vec<T>> {
    array(root, key, what)?.iter().map(|v| f(object(v, what)?)).collect()
}

/// Converts a parsed glTF JSON value to a [`Document`].
pub fn document(root: &Value) -> Result<Document> {
    let root = object(root, "glTF root is not an object")?;
    let asset = asset(opt_object(root, "asset", "asset is not an object")?.ok_or(invalid("asset is missing"))?)?;
    let lights = match extension(root, "KHR_lights_punctual")? {
        Some(ext) => each(ext, "lights", "KHR_lights_punctual.lights is not an array of objects", light)?,
        None => Vec::new(),
    };
    Ok(Document {
        asset,
        extensions_used: strings(root, "extensionsUsed", "extensionsUsed is not an array of strings")?,
        extensions_required: strings(root, "extensionsRequired", "extensionsRequired is not an array of strings")?,
        scene: opt_usize(root, "scene", "scene is not an index")?,
        scenes: each(root, "scenes", "scenes is not an array of objects", scene)?,
        nodes: each(root, "nodes", "nodes is not an array of objects", node)?,
        meshes: each(root, "meshes", "meshes is not an array of objects", mesh)?,
        skins: each(root, "skins", "skins is not an array of objects", skin)?,
        accessors: each(root, "accessors", "accessors is not an array of objects", accessor)?,
        buffer_views: each(root, "bufferViews", "bufferViews is not an array of objects", buffer_view)?,
        buffers: each(root, "buffers", "buffers is not an array of objects", buffer)?,
        materials: each(root, "materials", "materials is not an array of objects", material)?,
        textures: each(root, "textures", "textures is not an array of objects", texture)?,
        images: each(root, "images", "images is not an array of objects", image)?,
        samplers: each(root, "samplers", "samplers is not an array of objects", sampler)?,
        animations: each(root, "animations", "animations is not an array of objects", animation)?,
        cameras: each(root, "cameras", "cameras is not an array of objects", camera)?,
        lights,
        extra: extra(root, &["KHR_lights_punctual"])?,
    })
}

fn asset(o: &Object) -> Result<Asset> {
    Ok(Asset {
        version: opt_string(o, "version", "asset.version is not a string")?
            .ok_or(invalid("asset.version is missing"))?,
        min_version: opt_string(o, "minVersion", "asset.minVersion is not a string")?,
        generator: opt_string(o, "generator", "asset.generator is not a string")?,
        copyright: opt_string(o, "copyright", "asset.copyright is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn scene(o: &Object) -> Result<Scene> {
    Ok(Scene {
        name: opt_string(o, "name", "scene.name is not a string")?,
        nodes: indices(o, "nodes", "scene.nodes is not an array of indices")?,
        extra: extra(o, &[])?,
    })
}

fn node(o: &Object) -> Result<Node> {
    const WHAT: &str = "node transform is not an array of finite numbers of the right length";
    let matrix = opt_floats::<16>(o, "matrix", WHAT)?;
    let translation = opt_floats::<3>(o, "translation", WHAT)?;
    let rotation = opt_floats::<4>(o, "rotation", WHAT)?;
    let scale = opt_floats::<3>(o, "scale", WHAT)?;
    let transform = if let Some(m) = matrix {
        if translation.is_some() || rotation.is_some() || scale.is_some() {
            return Err(invalid("node has both a matrix and translation, rotation or scale"));
        }
        Transform::Matrix(m)
    } else {
        let rotation = rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let length = rotation.iter().map(|c| c * c).sum::<f32>().sqrt();
        if (length - 1.0).abs() > 1e-3 {
            return Err(invalid("node.rotation is not a unit quaternion"));
        }
        Transform::Trs { translation: translation.unwrap_or([0.0; 3]), rotation, scale: scale.unwrap_or([1.0; 3]) }
    };
    let light = match extension(o, "KHR_lights_punctual")? {
        Some(ext) => Some(req_usize(ext, "light", "node KHR_lights_punctual.light is not an index")?),
        None => None,
    };
    Ok(Node {
        name: opt_string(o, "name", "node.name is not a string")?,
        children: indices(o, "children", "node.children is not an array of indices")?,
        camera: opt_usize(o, "camera", "node.camera is not an index")?,
        mesh: opt_usize(o, "mesh", "node.mesh is not an index")?,
        skin: opt_usize(o, "skin", "node.skin is not an index")?,
        transform,
        weights: float_list(o, "weights", "node.weights is not an array of numbers")?,
        light,
        extra: extra(o, &["KHR_lights_punctual"])?,
    })
}

fn attributes(o: &Object, what: &'static str) -> Result<Vec<Attribute>> {
    o.iter().map(|(name, v)| Ok((name.to_owned(), usize_of(v, what)?))).collect()
}

fn mesh(o: &Object) -> Result<Mesh> {
    let primitives = each(o, "primitives", "mesh.primitives is not an array of objects", |p| {
        let attrs = opt_object(p, "attributes", "primitive.attributes is not an object")?
            .ok_or(invalid("primitive.attributes is missing"))?;
        let targets = array(p, "targets", "primitive.targets is not an array")?
            .iter()
            .map(|t| attributes(object(t, "morph target is not an object")?, "morph target attribute is not an index"))
            .collect::<Result<Vec<_>>>()?;
        let mode = match opt_uint(p, "mode", "primitive.mode is not an integer")? {
            Some(code) => Mode::from_code(code).ok_or(invalid("primitive.mode is not a known mode"))?,
            None => Mode::Triangles,
        };
        Ok(Primitive {
            attributes: attributes(attrs, "primitive attribute is not an index")?,
            indices: opt_usize(p, "indices", "primitive.indices is not an index")?,
            material: opt_usize(p, "material", "primitive.material is not an index")?,
            mode,
            targets,
            extra: extra(p, &[])?,
        })
    })?;
    Ok(Mesh {
        name: opt_string(o, "name", "mesh.name is not a string")?,
        primitives,
        weights: float_list(o, "weights", "mesh.weights is not an array of numbers")?,
        extra: extra(o, &[])?,
    })
}

fn skin(o: &Object) -> Result<Skin> {
    Ok(Skin {
        name: opt_string(o, "name", "skin.name is not a string")?,
        inverse_bind_matrices: opt_usize(o, "inverseBindMatrices", "skin.inverseBindMatrices is not an index")?,
        skeleton: opt_usize(o, "skeleton", "skin.skeleton is not an index")?,
        joints: indices(o, "joints", "skin.joints is not an array of indices")?,
        extra: extra(o, &[])?,
    })
}

fn component_type(o: &Object, what: &'static str) -> Result<ComponentType> {
    let code = opt_uint(o, "componentType", what)?.ok_or(invalid(what))?;
    ComponentType::from_code(code).ok_or(invalid(what))
}

fn bounds(o: &Object, key: &str, what: &'static str) -> Result<Vec<f64>> {
    array(o, key, what)?.iter().map(|v| v.as_f64().ok_or(invalid(what))).collect()
}

fn accessor(o: &Object) -> Result<Accessor> {
    let kind = o.get("type").and_then(Value::as_str).ok_or(invalid("accessor.type is missing or not a string"))?;
    let kind = ElementType::NAMES
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|&(_, k)| k)
        .ok_or(invalid("accessor.type is not a known type"))?;
    let sparse = match opt_object(o, "sparse", "accessor.sparse is not an object")? {
        Some(s) => {
            let idx = opt_object(s, "indices", "sparse.indices is not an object")?
                .ok_or(invalid("sparse.indices is missing"))?;
            let val = opt_object(s, "values", "sparse.values is not an object")?
                .ok_or(invalid("sparse.values is missing"))?;
            Some(Sparse {
                count: req_usize(s, "count", "sparse.count is missing or not an integer")?,
                indices_view: req_usize(idx, "bufferView", "sparse.indices.bufferView is missing or not an index")?,
                indices_offset: opt_usize(idx, "byteOffset", "sparse.indices.byteOffset is not an integer")?
                    .unwrap_or(0),
                indices_type: component_type(idx, "sparse.indices.componentType is missing or unknown")?,
                values_view: req_usize(val, "bufferView", "sparse.values.bufferView is missing or not an index")?,
                values_offset: opt_usize(val, "byteOffset", "sparse.values.byteOffset is not an integer")?.unwrap_or(0),
                extra: extra(s, &[])?,
            })
        }
        None => None,
    };
    let component_type = component_type(o, "accessor.componentType is missing or unknown")?;
    // Bounds of float accessors are `f32` values: keep them as the `f32` they denote, so a document
    // read, written (with `f32` digits) and read again compares equal.
    let narrow = |v: Vec<f64>| -> Vec<f64> {
        if component_type == ComponentType::F32 { v.into_iter().map(|x| f64::from(x as f32)).collect() } else { v }
    };
    Ok(Accessor {
        buffer_view: opt_usize(o, "bufferView", "accessor.bufferView is not an index")?,
        byte_offset: opt_usize(o, "byteOffset", "accessor.byteOffset is not an integer")?.unwrap_or(0),
        component_type,
        normalized: bool_or(o, "normalized", false, "accessor.normalized is not a boolean")?,
        count: req_usize(o, "count", "accessor.count is missing or not an integer")?,
        kind,
        min: narrow(bounds(o, "min", "accessor.min is not an array of numbers")?),
        max: narrow(bounds(o, "max", "accessor.max is not an array of numbers")?),
        sparse,
        name: opt_string(o, "name", "accessor.name is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn buffer_view(o: &Object) -> Result<BufferView> {
    let target = match opt_uint(o, "target", "bufferView.target is not an integer")? {
        None => None,
        Some(34962) => Some(Target::ArrayBuffer),
        Some(34963) => Some(Target::ElementArrayBuffer),
        Some(_) => return Err(invalid("bufferView.target is not a known target")),
    };
    Ok(BufferView {
        buffer: req_usize(o, "buffer", "bufferView.buffer is missing or not an index")?,
        byte_offset: opt_usize(o, "byteOffset", "bufferView.byteOffset is not an integer")?.unwrap_or(0),
        byte_length: req_usize(o, "byteLength", "bufferView.byteLength is missing or not an integer")?,
        byte_stride: opt_usize(o, "byteStride", "bufferView.byteStride is not an integer")?,
        target,
        name: opt_string(o, "name", "bufferView.name is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn buffer(o: &Object) -> Result<Buffer> {
    Ok(Buffer {
        uri: opt_string(o, "uri", "buffer.uri is not a string")?,
        byte_length: req_usize(o, "byteLength", "buffer.byteLength is missing or not an integer")?,
        name: opt_string(o, "name", "buffer.name is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn texture_transform(o: &Object) -> Result<TextureTransform> {
    const WHAT: &str = "KHR_texture_transform member is not a finite number or pair";
    Ok(TextureTransform {
        offset: floats_or(o, "offset", [0.0; 2], WHAT)?,
        rotation: f32_or(o, "rotation", 0.0, WHAT)?,
        scale: floats_or(o, "scale", [1.0; 2], WHAT)?,
        tex_coord: opt_u32(o, "texCoord", "KHR_texture_transform.texCoord is not an integer")?,
    })
}

/// A texture info; `scale_key` is `scale` for normal textures, `strength` for occlusion and `None`
/// for the others.
fn texture_info(o: &Object, key: &str, scale_key: Option<&str>) -> Result<Option<TextureInfo>> {
    let Some(t) = opt_object(o, key, "texture info is not an object")? else { return Ok(None) };
    let transform = extension(t, "KHR_texture_transform")?.map(texture_transform).transpose()?;
    Ok(Some(TextureInfo {
        index: req_usize(t, "index", "texture info index is missing or not an index")?,
        tex_coord: opt_u32(t, "texCoord", "texture info texCoord is not an integer")?.unwrap_or(0),
        scale: match scale_key {
            Some(k) => f32_or(t, k, 1.0, "texture info scale or strength is not a number")?,
            None => 1.0,
        },
        transform,
        extra: extra(t, &["KHR_texture_transform"])?,
    }))
}

/// A texture info without a scale.
fn tex(o: &Object, key: &str) -> Result<Option<TextureInfo>> {
    texture_info(o, key, None)
}

/// The typed material extensions, as written in `extensions`.
const MATERIAL_EXTENSIONS: [&str; 9] = [
    "KHR_materials_unlit",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_specular",
    "KHR_materials_clearcoat",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_diffuse_transmission",
];

fn material(o: &Object) -> Result<Material> {
    const FACTOR: &str = "material factor is not a finite number or has the wrong length";
    let pbr = match opt_object(o, "pbrMetallicRoughness", "material.pbrMetallicRoughness is not an object")? {
        Some(p) => Pbr {
            base_color_factor: floats_or(p, "baseColorFactor", [1.0; 4], FACTOR)?,
            base_color_texture: tex(p, "baseColorTexture")?,
            metallic_factor: f32_or(p, "metallicFactor", 1.0, FACTOR)?,
            roughness_factor: f32_or(p, "roughnessFactor", 1.0, FACTOR)?,
            metallic_roughness_texture: tex(p, "metallicRoughnessTexture")?,
            extra: extra(p, &[])?,
        },
        None => Pbr::default(),
    };
    let alpha_mode = match o.get("alphaMode") {
        None => AlphaMode::Opaque,
        Some(v) => match v.as_str() {
            Some("OPAQUE") => AlphaMode::Opaque,
            Some("MASK") => AlphaMode::Mask,
            Some("BLEND") => AlphaMode::Blend,
            _ => return Err(invalid("material.alphaMode is not OPAQUE, MASK or BLEND")),
        },
    };
    let ext = |name| extension(o, name);
    let specular = ext("KHR_materials_specular")?
        .map(|e| {
            Ok::<_, Error>(Specular {
                factor: f32_or(e, "specularFactor", 1.0, FACTOR)?,
                texture: tex(e, "specularTexture")?,
                color_factor: floats_or(e, "specularColorFactor", [1.0; 3], FACTOR)?,
                color_texture: tex(e, "specularColorTexture")?,
            })
        })
        .transpose()?;
    let clearcoat = ext("KHR_materials_clearcoat")?
        .map(|e| {
            Ok::<_, Error>(Clearcoat {
                factor: f32_or(e, "clearcoatFactor", 0.0, FACTOR)?,
                texture: tex(e, "clearcoatTexture")?,
                roughness_factor: f32_or(e, "clearcoatRoughnessFactor", 0.0, FACTOR)?,
                roughness_texture: tex(e, "clearcoatRoughnessTexture")?,
                normal_texture: texture_info(e, "clearcoatNormalTexture", Some("scale"))?,
            })
        })
        .transpose()?;
    let sheen = ext("KHR_materials_sheen")?
        .map(|e| {
            Ok::<_, Error>(Sheen {
                color_factor: floats_or(e, "sheenColorFactor", [0.0; 3], FACTOR)?,
                color_texture: tex(e, "sheenColorTexture")?,
                roughness_factor: f32_or(e, "sheenRoughnessFactor", 0.0, FACTOR)?,
                roughness_texture: tex(e, "sheenRoughnessTexture")?,
            })
        })
        .transpose()?;
    let transmission = ext("KHR_materials_transmission")?
        .map(|e| {
            Ok::<_, Error>(Transmission {
                factor: f32_or(e, "transmissionFactor", 0.0, FACTOR)?,
                texture: tex(e, "transmissionTexture")?,
            })
        })
        .transpose()?;
    let volume = ext("KHR_materials_volume")?
        .map(|e| {
            Ok::<_, Error>(Volume {
                thickness_factor: f32_or(e, "thicknessFactor", 0.0, FACTOR)?,
                thickness_texture: tex(e, "thicknessTexture")?,
                attenuation_distance: opt_f32(e, "attenuationDistance", FACTOR)?,
                attenuation_color: floats_or(e, "attenuationColor", [1.0; 3], FACTOR)?,
            })
        })
        .transpose()?;
    let diffuse_transmission = ext("KHR_materials_diffuse_transmission")?
        .map(|e| {
            Ok::<_, Error>(DiffuseTransmission {
                factor: f32_or(e, "diffuseTransmissionFactor", 0.0, FACTOR)?,
                texture: tex(e, "diffuseTransmissionTexture")?,
                color_factor: floats_or(e, "diffuseTransmissionColorFactor", [1.0; 3], FACTOR)?,
                color_texture: tex(e, "diffuseTransmissionColorTexture")?,
            })
        })
        .transpose()?;
    Ok(Material {
        name: opt_string(o, "name", "material.name is not a string")?,
        pbr,
        normal_texture: texture_info(o, "normalTexture", Some("scale"))?,
        occlusion_texture: texture_info(o, "occlusionTexture", Some("strength"))?,
        emissive_texture: tex(o, "emissiveTexture")?,
        emissive_factor: floats_or(o, "emissiveFactor", [0.0; 3], FACTOR)?,
        alpha_mode,
        alpha_cutoff: f32_or(o, "alphaCutoff", 0.5, FACTOR)?,
        double_sided: bool_or(o, "doubleSided", false, "material.doubleSided is not a boolean")?,
        unlit: ext("KHR_materials_unlit")?.is_some(),
        emissive_strength: ext("KHR_materials_emissive_strength")?
            .map(|e| f32_or(e, "emissiveStrength", 1.0, FACTOR))
            .transpose()?,
        ior: ext("KHR_materials_ior")?.map(|e| f32_or(e, "ior", 1.5, FACTOR)).transpose()?,
        specular,
        clearcoat,
        sheen,
        transmission,
        volume,
        diffuse_transmission,
        extra: extra(o, &MATERIAL_EXTENSIONS)?,
    })
}

fn texture(o: &Object) -> Result<Texture> {
    let basisu_source = match extension(o, "KHR_texture_basisu")? {
        Some(e) => Some(req_usize(e, "source", "KHR_texture_basisu.source is missing or not an index")?),
        None => None,
    };
    Ok(Texture {
        sampler: opt_usize(o, "sampler", "texture.sampler is not an index")?,
        source: opt_usize(o, "source", "texture.source is not an index")?,
        basisu_source,
        name: opt_string(o, "name", "texture.name is not a string")?,
        extra: extra(o, &["KHR_texture_basisu"])?,
    })
}

fn image(o: &Object) -> Result<Image> {
    Ok(Image {
        uri: opt_string(o, "uri", "image.uri is not a string")?,
        mime_type: opt_string(o, "mimeType", "image.mimeType is not a string")?,
        buffer_view: opt_usize(o, "bufferView", "image.bufferView is not an index")?,
        name: opt_string(o, "name", "image.name is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn sampler(o: &Object) -> Result<Sampler> {
    const WHAT: &str = "sampler filter or wrap mode is not a known value";
    let filter = |key, allowed: &[u32]| -> Result<Option<u32>> {
        let v = opt_u32(o, key, WHAT)?;
        if v.is_some_and(|v| !allowed.contains(&v)) { Err(invalid(WHAT)) } else { Ok(v) }
    };
    let wrap = |key| -> Result<u32> {
        let v = opt_u32(o, key, WHAT)?.unwrap_or(super::doc::REPEAT);
        if [33071, 33648, 10497].contains(&v) { Ok(v) } else { Err(invalid(WHAT)) }
    };
    Ok(Sampler {
        mag_filter: filter("magFilter", &[9728, 9729])?,
        min_filter: filter("minFilter", &[9728, 9729, 9984, 9985, 9986, 9987])?,
        wrap_s: wrap("wrapS")?,
        wrap_t: wrap("wrapT")?,
        name: opt_string(o, "name", "sampler.name is not a string")?,
        extra: extra(o, &[])?,
    })
}

fn animation(o: &Object) -> Result<Animation> {
    let channels = each(o, "channels", "animation.channels is not an array of objects", |c| {
        let t =
            opt_object(c, "target", "channel.target is not an object")?.ok_or(invalid("channel.target is missing"))?;
        let path = t.get("path").and_then(Value::as_str).ok_or(invalid("channel.target.path is missing"))?;
        let path = Path::NAMES
            .iter()
            .find(|(name, _)| *name == path)
            .map(|&(_, p)| p)
            .ok_or(invalid("channel.target.path is not a known path"))?;
        Ok(Channel {
            sampler: req_usize(c, "sampler", "channel.sampler is missing or not an index")?,
            node: opt_usize(t, "node", "channel.target.node is not an index")?,
            path,
            extra: extra(c, &[])?,
            target_extra: extra(t, &[])?,
        })
    })?;
    let samplers = each(o, "samplers", "animation.samplers is not an array of objects", |s| {
        let interpolation = match s.get("interpolation") {
            None => Interpolation::Linear,
            Some(v) => match v.as_str() {
                Some("LINEAR") => Interpolation::Linear,
                Some("STEP") => Interpolation::Step,
                Some("CUBICSPLINE") => Interpolation::CubicSpline,
                _ => return Err(invalid("animation sampler interpolation is not a known value")),
            },
        };
        Ok(AnimationSampler {
            input: req_usize(s, "input", "animation sampler input is missing or not an index")?,
            output: req_usize(s, "output", "animation sampler output is missing or not an index")?,
            interpolation,
            extra: extra(s, &[])?,
        })
    })?;
    Ok(Animation {
        name: opt_string(o, "name", "animation.name is not a string")?,
        channels,
        samplers,
        extra: extra(o, &[])?,
    })
}

fn camera(o: &Object) -> Result<Camera> {
    const WHAT: &str = "camera projection member is missing or not a finite number";
    let kind = o.get("type").and_then(Value::as_str).ok_or(invalid("camera.type is missing or not a string"))?;
    let projection = match kind {
        "perspective" => {
            let p = opt_object(o, "perspective", "camera.perspective is not an object")?
                .ok_or(invalid("camera.perspective is missing"))?;
            Projection::Perspective {
                aspect_ratio: opt_f32(p, "aspectRatio", WHAT)?,
                yfov: opt_f32(p, "yfov", WHAT)?.ok_or(invalid(WHAT))?,
                zfar: opt_f32(p, "zfar", WHAT)?,
                znear: opt_f32(p, "znear", WHAT)?.ok_or(invalid(WHAT))?,
            }
        }
        "orthographic" => {
            let p = opt_object(o, "orthographic", "camera.orthographic is not an object")?
                .ok_or(invalid("camera.orthographic is missing"))?;
            let get = |key| opt_f32(p, key, WHAT)?.ok_or(invalid(WHAT));
            Projection::Orthographic {
                xmag: get("xmag")?,
                ymag: get("ymag")?,
                zfar: get("zfar")?,
                znear: get("znear")?,
            }
        }
        _ => return Err(invalid("camera.type is not perspective or orthographic")),
    };
    Ok(Camera { name: opt_string(o, "name", "camera.name is not a string")?, projection, extra: extra(o, &[])? })
}

fn light(o: &Object) -> Result<Light> {
    const WHAT: &str = "light member is not a finite number or has the wrong length";
    let kind = match o.get("type").and_then(Value::as_str) {
        Some("directional") => LightKind::Directional,
        Some("point") => LightKind::Point,
        Some("spot") => {
            let s = opt_object(o, "spot", "light.spot is not an object")?.unwrap_or(&EMPTY);
            LightKind::Spot {
                inner_cone_angle: f32_or(s, "innerConeAngle", 0.0, WHAT)?,
                outer_cone_angle: f32_or(s, "outerConeAngle", std::f32::consts::FRAC_PI_4, WHAT)?,
            }
        }
        _ => return Err(invalid("light.type is not directional, point or spot")),
    };
    Ok(Light {
        name: opt_string(o, "name", "light.name is not a string")?,
        color: floats_or(o, "color", [1.0; 3], WHAT)?,
        intensity: f32_or(o, "intensity", 1.0, WHAT)?,
        kind,
        range: opt_f32(o, "range", WHAT)?,
        extra: extra(o, &[])?,
    })
}
