//! Cross-reference validation of a [`Document`] and its buffer data `[IO-10]` `[ROB-01]`.
//!
//! Run by the reader on every file and by the writers before anything is written, so a document
//! this crate accepts or produces has: indices in range, buffer views inside their buffers,
//! accessors inside their views with aligned offsets, sparse indices strictly increasing and in
//! range, attribute types the specification (and `KHR_mesh_quantization`, when used) allows, equal
//! vertex counts per primitive, consistent morph target counts, a node forest without cycles or
//! shared children no deeper than [`Limits::max_depth`], well-formed skins, animation samplers whose
//! output counts match their inputs, and sane camera and light parameters.

use super::doc::{
    Accessor, AnimationSampler, Attribute, ComponentType, Document, ElementType, Interpolation, LightKind, Path,
    Projection, SUPPORTED_EXTENSIONS, Transform,
};
use super::parse::invalid;
use crate::error::{Error, Format, Result};
use crate::image::Limits;

/// The element at `index`, or an [`Error::Invalid`] naming `what`.
fn get<'a, T>(items: &'a [T], index: usize, what: &'static str) -> Result<&'a T> {
    items.get(index).ok_or(invalid(what))
}

/// Fails unless `index < len`.
fn check(index: usize, len: usize, what: &'static str) -> Result<()> {
    if index < len { Ok(()) } else { Err(invalid(what)) }
}

/// Checks a whole document against its buffer data (`buffers[i]` holds buffer `i`, at least
/// `byte_length` bytes).
pub fn validate(doc: &Document, buffers: &[Vec<u8>], limits: &Limits) -> Result<()> {
    asset_and_extensions(doc)?;
    buffers_and_views(doc, buffers)?;
    for accessor in &doc.accessors {
        check_accessor(doc, buffers, accessor, limits)?;
    }
    textures_and_materials(doc)?;
    meshes(doc)?;
    nodes_and_scenes(doc, limits)?;
    skins(doc)?;
    animations(doc)?;
    cameras_and_lights(doc)
}

fn asset_and_extensions(doc: &Document) -> Result<()> {
    let major_minor = |v: &str| -> Option<(u32, u32)> {
        let (major, minor) = v.split_once('.')?;
        Some((major.parse().ok()?, minor.parse().ok()?))
    };
    let (major, _) = major_minor(&doc.asset.version).ok_or(invalid("asset.version is not MAJOR.MINOR"))?;
    if major != 2 {
        return Err(Error::Unsupported { format: Format::Gltf, what: "glTF major version other than 2" });
    }
    if let Some(min) = &doc.asset.min_version {
        let (major, minor) = major_minor(min).ok_or(invalid("asset.minVersion is not MAJOR.MINOR"))?;
        if (major, minor) > (2, 0) {
            return Err(Error::Unsupported { format: Format::Gltf, what: "asset.minVersion above 2.0" });
        }
    }
    for name in &doc.extensions_required {
        if !SUPPORTED_EXTENSIONS.contains(&name.as_str()) {
            return Err(Error::Unsupported { format: Format::Gltf, what: "a required glTF extension" });
        }
        if !doc.extensions_used.contains(name) {
            return Err(invalid("extensionsRequired names an extension missing from extensionsUsed"));
        }
    }
    Ok(())
}

fn buffers_and_views(doc: &Document, buffers: &[Vec<u8>]) -> Result<()> {
    if buffers.len() != doc.buffers.len() {
        return Err(invalid("buffer data count differs from the buffers array"));
    }
    for (buffer, data) in doc.buffers.iter().zip(buffers) {
        if buffer.byte_length == 0 {
            return Err(invalid("buffer.byteLength is 0"));
        }
        if data.len() < buffer.byte_length {
            return Err(invalid("buffer data is shorter than buffer.byteLength"));
        }
    }
    for view in &doc.buffer_views {
        let buffer = get(&doc.buffers, view.buffer, "bufferView.buffer is out of range")?;
        if view.byte_length == 0 {
            return Err(invalid("bufferView.byteLength is 0"));
        }
        let end = view.byte_offset.checked_add(view.byte_length).ok_or(invalid("bufferView range overflows"))?;
        if end > buffer.byte_length {
            return Err(invalid("bufferView reaches past the end of its buffer"));
        }
        if let Some(stride) = view.byte_stride
            && (!(4..=252).contains(&stride) || !stride.is_multiple_of(4))
        {
            return Err(invalid("bufferView.byteStride is not a multiple of 4 between 4 and 252"));
        }
    }
    Ok(())
}

/// The bytes of buffer view `index`, with its offset applied.
pub(crate) fn view_bytes<'a>(doc: &Document, buffers: &'a [Vec<u8>], index: usize) -> Result<&'a [u8]> {
    let view = get(&doc.buffer_views, index, "bufferView index is out of range")?;
    let data = get(buffers, view.buffer, "bufferView.buffer is out of range")?;
    let end = view.byte_offset.checked_add(view.byte_length).ok_or(invalid("bufferView range overflows"))?;
    data.get(view.byte_offset..end).ok_or(invalid("bufferView reaches past the end of its buffer"))
}

/// Reads sparse index `i` from the tightly packed indices `bytes`.
pub(crate) fn sparse_index(bytes: &[u8], kind: ComponentType, i: usize) -> Result<usize> {
    let size = kind.size();
    let at = i * size;
    let b = bytes.get(at..at + size).ok_or(invalid("sparse indices reach past their bufferView"))?;
    Ok(match *b {
        [x] => usize::from(x),
        [x, y] => usize::from(u16::from_le_bytes([x, y])),
        [x, y, z, w] => u32::from_le_bytes([x, y, z, w]) as usize,
        _ => return Err(invalid("sparse.indices.componentType is not an unsigned integer type")),
    })
}

fn check_accessor(doc: &Document, buffers: &[Vec<u8>], a: &Accessor, limits: &Limits) -> Result<()> {
    if a.count == 0 {
        return Err(invalid("accessor.count is 0"));
    }
    let components = a.kind.components();
    limits.check_elements(Format::Gltf, "accessor element count", a.count as u64)?;
    limits.check_bytes(
        Format::Gltf,
        "decoded accessor size",
        (a.count as u64).saturating_mul(components as u64 * 4),
    )?;
    if a.normalized && matches!(a.component_type, ComponentType::F32 | ComponentType::U32) {
        return Err(invalid("accessor.normalized is true for a float or u32 accessor"));
    }
    if !(a.min.is_empty() || a.min.len() == components) || !(a.max.is_empty() || a.max.len() == components) {
        return Err(invalid("accessor.min or max has the wrong number of components"));
    }
    if a.min.iter().chain(&a.max).any(|v| !v.is_finite()) {
        return Err(invalid("accessor.min or max is not finite"));
    }
    let size = a.component_type.size();
    let elem = a.kind.size(a.component_type);
    match a.buffer_view {
        Some(index) => {
            let view = get(&doc.buffer_views, index, "accessor.bufferView is out of range")?;
            let stride = view.byte_stride.unwrap_or(elem);
            if stride < elem {
                return Err(invalid("bufferView.byteStride is smaller than the accessor element"));
            }
            if !a.byte_offset.is_multiple_of(size)
                || !(view.byte_offset.wrapping_add(a.byte_offset)).is_multiple_of(size)
            {
                return Err(invalid("accessor data is not aligned to its component size"));
            }
            let need = stride
                .checked_mul(a.count - 1)
                .and_then(|n| n.checked_add(elem))
                .and_then(|n| n.checked_add(a.byte_offset))
                .ok_or(invalid("accessor range overflows"))?;
            if need > view.byte_length {
                return Err(invalid("accessor reaches past the end of its bufferView"));
            }
        }
        None if a.byte_offset != 0 => return Err(invalid("accessor.byteOffset without a bufferView")),
        None => {}
    }
    let Some(sparse) = &a.sparse else { return Ok(()) };
    if sparse.count == 0 || sparse.count > a.count {
        return Err(invalid("sparse.count is 0 or above accessor.count"));
    }
    if !matches!(sparse.indices_type, ComponentType::U8 | ComponentType::U16 | ComponentType::U32) {
        return Err(invalid("sparse.indices.componentType is not an unsigned integer type"));
    }
    let index_size = sparse.indices_type.size();
    for (view, offset, item, align) in [
        (sparse.indices_view, sparse.indices_offset, index_size, index_size),
        (sparse.values_view, sparse.values_offset, elem, size),
    ] {
        let v = get(&doc.buffer_views, view, "sparse bufferView is out of range")?;
        if v.byte_stride.is_some() {
            return Err(invalid("sparse bufferView has a byteStride"));
        }
        if !v.byte_offset.wrapping_add(offset).is_multiple_of(align) {
            return Err(invalid("sparse data is not aligned to its component size"));
        }
        let need = item
            .checked_mul(sparse.count)
            .and_then(|n| n.checked_add(offset))
            .ok_or(invalid("sparse range overflows"))?;
        if need > v.byte_length {
            return Err(invalid("sparse data reaches past the end of its bufferView"));
        }
    }
    let bytes = view_bytes(doc, buffers, sparse.indices_view)?;
    let bytes = bytes.get(sparse.indices_offset..).ok_or(invalid("sparse indices reach past their bufferView"))?;
    let mut previous = None;
    for i in 0..sparse.count {
        let index = sparse_index(bytes, sparse.indices_type, i)?;
        if index >= a.count || previous.is_some_and(|p| index <= p) {
            return Err(invalid("sparse indices are not strictly increasing below accessor.count"));
        }
        previous = Some(index);
    }
    Ok(())
}

fn textures_and_materials(doc: &Document) -> Result<()> {
    for image in &doc.images {
        match (&image.uri, image.buffer_view) {
            (Some(_), None) => {}
            (None, Some(view)) => {
                check(view, doc.buffer_views.len(), "image.bufferView is out of range")?;
                if image.mime_type.is_none() {
                    return Err(invalid("image in a bufferView has no mimeType"));
                }
            }
            _ => return Err(invalid("image needs exactly one of uri and bufferView")),
        }
    }
    for texture in &doc.textures {
        for (index, len, what) in [
            (texture.sampler, doc.samplers.len(), "texture.sampler is out of range"),
            (texture.source, doc.images.len(), "texture.source is out of range"),
            (texture.basisu_source, doc.images.len(), "KHR_texture_basisu.source is out of range"),
        ] {
            if let Some(i) = index {
                check(i, len, what)?;
            }
        }
    }
    for material in &doc.materials {
        for info in material.texture_infos() {
            check(info.index, doc.textures.len(), "material texture index is out of range")?;
        }
        if material.alpha_cutoff < 0.0 {
            return Err(invalid("material.alphaCutoff is negative"));
        }
    }
    Ok(())
}

/// The semantic of an attribute name: `TEXCOORD_1` is `TEXCOORD`.
fn semantic(name: &str) -> &str {
    match name.rsplit_once('_') {
        Some((base, set)) if !base.is_empty() && !set.is_empty() && set.bytes().all(|b| b.is_ascii_digit()) => base,
        _ => name,
    }
}

/// Whether accessor `a` may hold attribute `name` of a primitive (`target` false) or of a morph
/// target (`target` true).
fn attribute_type_ok(name: &str, a: &Accessor, quantized: bool, target: bool) -> bool {
    use ComponentType::{F32, I8, I16, U8, U16};
    use ElementType::{Vec2, Vec3, Vec4};
    let (c, n, k) = (a.component_type, a.normalized, a.kind);
    let f32_only = c == F32;
    let signed_norm = quantized && matches!(c, I8 | I16) && n;
    let unorm = matches!(c, U8 | U16) && n;
    match (semantic(name), target) {
        ("POSITION", false) => k == Vec3 && (f32_only || quantized && matches!(c, I8 | U8 | I16 | U16)),
        ("POSITION", true) => k == Vec3 && (f32_only || quantized && matches!(c, I8 | I16)),
        ("NORMAL", _) | ("TANGENT", true) => k == Vec3 && (f32_only || signed_norm),
        ("TANGENT", false) => k == Vec4 && (f32_only || signed_norm),
        ("TEXCOORD", false) => k == Vec2 && (f32_only || unorm || quantized && matches!(c, I8 | U8 | I16 | U16)),
        ("TEXCOORD", true) => k == Vec2 && (f32_only || quantized && matches!(c, I8 | I16)),
        ("COLOR", false) => matches!(k, Vec3 | Vec4) && (f32_only || unorm),
        ("COLOR", true) => matches!(k, Vec3 | Vec4) && (f32_only || quantized && matches!(c, I8 | I16) && n),
        ("JOINTS", false) => k == Vec4 && matches!(c, U8 | U16) && !n,
        ("WEIGHTS", false) => k == Vec4 && (f32_only || unorm),
        ("JOINTS" | "WEIGHTS", true) => false,
        // Application-specific (`_NAME`) and unknown attributes are not typed by the specification.
        _ => true,
    }
}

/// Checks a list of attributes and returns their common element count.
fn attribute_set(doc: &Document, attributes: &[Attribute], quantized: bool, target: bool) -> Result<Option<usize>> {
    let mut count = None;
    for (name, index) in attributes {
        let a = get(&doc.accessors, *index, "attribute accessor is out of range")?;
        if !attribute_type_ok(name, a, quantized, target) {
            return Err(invalid("attribute accessor has a type the specification does not allow"));
        }
        if count.is_some_and(|c| c != a.count) {
            return Err(invalid("attributes of one primitive have different counts"));
        }
        count = Some(a.count);
    }
    Ok(count)
}

fn meshes(doc: &Document) -> Result<()> {
    let quantized = doc.extensions_used.iter().any(|e| e == "KHR_mesh_quantization");
    for mesh in &doc.meshes {
        let first = mesh.primitives.first().ok_or(invalid("mesh has no primitives"))?;
        let targets = first.targets.len();
        for p in &mesh.primitives {
            if p.attributes.is_empty() {
                return Err(invalid("primitive has no attributes"));
            }
            let count = attribute_set(doc, &p.attributes, quantized, false)?;
            if let Some(index) = p.indices {
                let a = get(&doc.accessors, index, "primitive.indices is out of range")?;
                if a.kind != ElementType::Scalar
                    || a.normalized
                    || !matches!(a.component_type, ComponentType::U8 | ComponentType::U16 | ComponentType::U32)
                {
                    return Err(invalid("primitive.indices is not an unsigned integer scalar accessor"));
                }
            }
            if let Some(m) = p.material {
                check(m, doc.materials.len(), "primitive.material is out of range")?;
            }
            if p.targets.len() != targets {
                return Err(invalid("primitives of one mesh have different morph target counts"));
            }
            for target in &p.targets {
                if target.is_empty() {
                    return Err(invalid("morph target has no attributes"));
                }
                if attribute_set(doc, target, quantized, true)? != count {
                    return Err(invalid("morph target count differs from the primitive's vertex count"));
                }
            }
        }
        if !mesh.weights.is_empty() && mesh.weights.len() != targets {
            return Err(invalid("mesh.weights length differs from the morph target count"));
        }
    }
    Ok(())
}

/// The morph target count of mesh `index`.
fn target_count(doc: &Document, index: usize) -> usize {
    doc.meshes.get(index).and_then(|m| m.primitives.first()).map_or(0, |p| p.targets.len())
}

fn nodes_and_scenes(doc: &Document, limits: &Limits) -> Result<()> {
    let n = doc.nodes.len();
    let mut parent = vec![usize::MAX; n];
    for (i, node) in doc.nodes.iter().enumerate() {
        for &child in &node.children {
            check(child, n, "node.children is out of range")?;
            if parent[child] != usize::MAX || child == i {
                return Err(invalid("node is the child of two parents or of itself"));
            }
            parent[child] = i;
        }
        for (index, len, what) in [
            (node.camera, doc.cameras.len(), "node.camera is out of range"),
            (node.mesh, doc.meshes.len(), "node.mesh is out of range"),
            (node.skin, doc.skins.len(), "node.skin is out of range"),
            (node.light, doc.lights.len(), "node KHR_lights_punctual.light is out of range"),
        ] {
            if let Some(i) = index {
                check(i, len, what)?;
            }
        }
        if node.skin.is_some() && node.mesh.is_none() {
            return Err(invalid("node has a skin but no mesh"));
        }
        if !node.weights.is_empty() && node.mesh.is_none_or(|m| target_count(doc, m) != node.weights.len()) {
            return Err(invalid("node.weights length differs from its mesh's morph target count"));
        }
        if let Transform::Matrix(m) = node.transform
            && m.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("node.matrix is not finite"));
        }
    }
    // Depth of every node, walking up to a node of known depth: linear time, no recursion, and a
    // walk longer than the node count is a cycle.
    let mut depth = vec![u32::MAX; n];
    let mut path = Vec::new();
    for start in 0..n {
        path.clear();
        let mut at = start;
        let base = loop {
            if depth[at] != u32::MAX {
                break depth[at];
            }
            if path.len() > n {
                return Err(invalid("the node hierarchy has a cycle"));
            }
            path.push(at);
            match parent[at] {
                usize::MAX => break 0,
                p => at = p,
            }
        };
        // A root on the path gets depth 1; the nodes below it count up from there.
        for (k, &node) in path.iter().rev().enumerate() {
            let d = base.saturating_add(u32::try_from(k).unwrap_or(u32::MAX)).saturating_add(1);
            if d > limits.max_depth {
                return Err(Error::Limit {
                    format: Format::Gltf,
                    what: "node hierarchy depth",
                    limit: u64::from(limits.max_depth),
                    got: u64::from(d),
                });
            }
            depth[node] = d;
        }
    }
    if let Some(s) = doc.scene {
        check(s, doc.scenes.len(), "scene is out of range")?;
    }
    let mut seen = vec![usize::MAX; n];
    for (s, scene) in doc.scenes.iter().enumerate() {
        for &root in &scene.nodes {
            check(root, n, "scene.nodes is out of range")?;
            if parent[root] != usize::MAX || seen[root] == s {
                return Err(invalid("scene.nodes holds a node that is not a root or is listed twice"));
            }
            seen[root] = s;
        }
    }
    Ok(())
}

fn skins(doc: &Document) -> Result<()> {
    let mut seen = vec![usize::MAX; doc.nodes.len()];
    for (s, skin) in doc.skins.iter().enumerate() {
        if skin.joints.is_empty() {
            return Err(invalid("skin has no joints"));
        }
        for &joint in &skin.joints {
            check(joint, doc.nodes.len(), "skin.joints is out of range")?;
            if seen[joint] == s {
                return Err(invalid("skin.joints lists a node twice"));
            }
            seen[joint] = s;
        }
        if let Some(root) = skin.skeleton {
            check(root, doc.nodes.len(), "skin.skeleton is out of range")?;
        }
        if let Some(index) = skin.inverse_bind_matrices {
            let a = get(&doc.accessors, index, "skin.inverseBindMatrices is out of range")?;
            if a.kind != ElementType::Mat4 || a.component_type != ComponentType::F32 || a.count < skin.joints.len() {
                return Err(invalid("skin.inverseBindMatrices is not a float MAT4 accessor with one matrix per joint"));
            }
        }
    }
    Ok(())
}

/// Whether `a` is a float, or a normalized integer when `ints` allows it.
fn float_or_normalized(a: &Accessor, ints: bool) -> bool {
    a.component_type == ComponentType::F32 || ints && a.normalized
}

fn animations(doc: &Document) -> Result<()> {
    for animation in &doc.animations {
        if animation.channels.is_empty() || animation.samplers.is_empty() {
            return Err(invalid("animation has no channels or no samplers"));
        }
        for sampler in &animation.samplers {
            let input = get(&doc.accessors, sampler.input, "animation sampler input is out of range")?;
            if input.kind != ElementType::Scalar || input.component_type != ComponentType::F32 {
                return Err(invalid("animation sampler input is not a float scalar accessor"));
            }
            check(sampler.output, doc.accessors.len(), "animation sampler output is out of range")?;
        }
        for channel in &animation.channels {
            let sampler: &AnimationSampler =
                get(&animation.samplers, channel.sampler, "channel.sampler is out of range")?;
            let node = match channel.node {
                Some(node) => Some(get(&doc.nodes, node, "channel.target.node is out of range")?),
                None if channel.path == Path::Pointer => None,
                None => return Err(invalid("channel.target.node is missing")),
            };
            let input = get(&doc.accessors, sampler.input, "animation sampler input is out of range")?;
            let output = get(&doc.accessors, sampler.output, "animation sampler output is out of range")?;
            let per_key = match channel.path {
                Path::Translation | Path::Scale => {
                    if output.kind != ElementType::Vec3 || output.component_type != ComponentType::F32 {
                        return Err(invalid("translation or scale output is not a float VEC3 accessor"));
                    }
                    1
                }
                Path::Rotation => {
                    if output.kind != ElementType::Vec4 || !float_or_normalized(output, true) {
                        return Err(invalid("rotation output is not a float or normalized VEC4 accessor"));
                    }
                    1
                }
                Path::Weights => {
                    let targets = node.and_then(|n| n.mesh).map_or(0, |m| target_count(doc, m));
                    if targets == 0 || output.kind != ElementType::Scalar || !float_or_normalized(output, true) {
                        return Err(invalid(
                            "weights channel targets a node without morph targets or has a bad output",
                        ));
                    }
                    targets
                }
                Path::Pointer => continue,
            };
            let per_key = if sampler.interpolation == Interpolation::CubicSpline { per_key * 3 } else { per_key };
            if input.count.checked_mul(per_key) != Some(output.count) {
                return Err(invalid("animation sampler output count does not match its input count"));
            }
        }
    }
    Ok(())
}

fn cameras_and_lights(doc: &Document) -> Result<()> {
    for camera in &doc.cameras {
        let ok = match camera.projection {
            Projection::Perspective { aspect_ratio, yfov, zfar, znear } => {
                yfov > 0.0 && znear > 0.0 && aspect_ratio.is_none_or(|a| a > 0.0) && zfar.is_none_or(|f| f > znear)
            }
            Projection::Orthographic { xmag, ymag, zfar, znear } => {
                xmag != 0.0 && ymag != 0.0 && znear >= 0.0 && zfar > znear
            }
        };
        if !ok {
            return Err(invalid("camera projection parameters are out of range"));
        }
    }
    for light in &doc.lights {
        let cone_ok = match light.kind {
            LightKind::Spot { inner_cone_angle: inner, outer_cone_angle: outer } => {
                inner >= 0.0 && inner < outer && outer <= std::f32::consts::FRAC_PI_2
            }
            LightKind::Directional | LightKind::Point => true,
        };
        if !cone_ok || light.intensity < 0.0 || light.range.is_some_and(|r| r <= 0.0) {
            return Err(invalid("light parameters are out of range"));
        }
    }
    Ok(())
}
