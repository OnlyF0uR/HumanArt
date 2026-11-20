//! [`Document`] to glTF JSON `[IO-10]`.
//!
//! Members equal to their specification default are left out, floats are written with the shortest
//! digits that read back as the same `f32`, and `extensionsUsed` lists every extension the document
//! writes (typed or kept in an [`Extra`]) plus the ones it already named, sorted. The output is
//! deterministic: the same document gives the same bytes.

use super::doc::{
    Accessor, AlphaMode, Animation, Buffer, BufferView, Camera, ComponentType, Document, Extra, Image, Interpolation,
    Light, LightKind, Material, Mesh, Node, Projection, REPEAT, Sampler, Scene, Skin, Texture, TextureInfo, Transform,
};
use crate::error::Result;
use crate::json::{JsonWriter, Sink, Style};

/// A sink that keeps nothing: the first pass only collects extension names.
struct Discard;

impl Sink for Discard {
    fn push_str(&mut self, _text: &str) {}
}

/// The `extensionsUsed` that [`to_json`] writes for `doc`: the names it already lists and requires,
/// plus every extension present in it (typed or kept in an [`Extra`]), sorted and without
/// duplicates.
pub fn used_extensions(doc: &Document) -> Result<Vec<String>> {
    let mut used: Vec<String> = doc.extensions_used.clone();
    used.extend(doc.extensions_required.iter().cloned());
    let mut sink = Discard;
    let mut pass = Out { w: JsonWriter::new(&mut sink, Style::COMPACT), used: &mut used, written: Vec::new() };
    pass.document(doc, &[])?;
    pass.w.finish()?;
    used.sort();
    used.dedup();
    Ok(used)
}

/// Writes `doc` as glTF JSON text in `style`.
pub fn to_json(doc: &Document, style: Style) -> Result<Vec<u8>> {
    let used = used_extensions(doc)?;
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    let mut pass = Out { w: JsonWriter::new(&mut out, style), used: &mut scratch, written: Vec::new() };
    pass.document(doc, &used)?;
    pass.w.finish()?;
    Ok(out)
}

/// One writing pass.
struct Out<'a, 'u, S: Sink> {
    w: JsonWriter<'a, S>,
    /// Every extension name written so far.
    used: &'u mut Vec<String>,
    /// The typed extensions written into the open `extensions` object.
    written: Vec<&'static str>,
}

impl<S: Sink> Out<'_, '_, S> {
    fn note(&mut self, name: &str) {
        if !self.used.iter().any(|u| u == name) {
            self.used.push(name.to_owned());
        }
    }

    fn string(&mut self, key: &str, value: &str) -> Result<()> {
        self.w.key(key)?.string(value)?;
        Ok(())
    }

    fn opt_string(&mut self, key: &str, value: Option<&String>) -> Result<()> {
        match value {
            Some(v) => self.string(key, v),
            None => Ok(()),
        }
    }

    fn name(&mut self, name: Option<&String>) -> Result<()> {
        self.opt_string("name", name)
    }

    fn uint(&mut self, key: &str, value: usize) -> Result<()> {
        self.w.key(key)?.u64(value as u64)?;
        Ok(())
    }

    fn opt_uint(&mut self, key: &str, value: Option<usize>) -> Result<()> {
        match value {
            Some(v) => self.uint(key, v),
            None => Ok(()),
        }
    }

    fn uint_unless(&mut self, key: &str, value: usize, default: usize) -> Result<()> {
        if value == default { Ok(()) } else { self.uint(key, value) }
    }

    fn float(&mut self, key: &str, value: f32) -> Result<()> {
        self.w.key(key)?.f32(value)?;
        Ok(())
    }

    fn float_unless(&mut self, key: &str, value: f32, default: f32) -> Result<()> {
        if value == default { Ok(()) } else { self.float(key, value) }
    }

    fn floats(&mut self, key: &str, values: &[f32]) -> Result<()> {
        self.w.key(key)?.begin_array_inline()?;
        for &v in values {
            self.w.f32(v)?;
        }
        self.w.end_array()?;
        Ok(())
    }

    fn floats_unless(&mut self, key: &str, values: &[f32], default: &[f32]) -> Result<()> {
        if values == default { Ok(()) } else { self.floats(key, values) }
    }

    fn indices(&mut self, key: &str, values: &[usize]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        self.w.key(key)?.begin_array_inline()?;
        for &v in values {
            self.w.u64(v as u64)?;
        }
        self.w.end_array()?;
        Ok(())
    }

    fn strings(&mut self, key: &str, values: &[String]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        self.w.key(key)?.begin_array()?;
        for v in values {
            self.w.string(v)?;
        }
        self.w.end_array()?;
        Ok(())
    }

    /// Opens the object of typed extension `name` inside an open `extensions` object.
    fn begin_ext(&mut self, name: &'static str) -> Result<()> {
        self.note(name);
        self.written.push(name);
        self.w.key(name)?.begin_object()?;
        Ok(())
    }

    /// Writes `extensions` (typed ones through `typed`, when `has_typed`, then the kept ones) and
    /// `extras`.
    fn extra_with(
        &mut self,
        extra: &Extra,
        has_typed: bool,
        typed: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        if has_typed || !extra.extensions.is_empty() {
            self.w.key("extensions")?.begin_object()?;
            let outer = std::mem::take(&mut self.written);
            typed(self)?;
            for (name, value) in &extra.extensions {
                if self.written.contains(&name) {
                    continue;
                }
                self.note(name);
                self.w.key(name)?.value(value)?;
            }
            self.written = outer;
            self.w.end_object()?;
        }
        if let Some(extras) = &extra.extras {
            self.w.key("extras")?.value(extras)?;
        }
        Ok(())
    }

    fn extra(&mut self, extra: &Extra) -> Result<()> {
        self.extra_with(extra, false, |_| Ok(()))
    }

    /// Writes the objects of a top-level array, if there are any.
    fn list<T>(&mut self, key: &str, items: &[T], mut f: impl FnMut(&mut Self, &T) -> Result<()>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.w.key(key)?.begin_array()?;
        for item in items {
            self.w.begin_object()?;
            f(self, item)?;
            self.w.end_object()?;
        }
        self.w.end_array()?;
        Ok(())
    }

    fn document(&mut self, doc: &Document, used: &[String]) -> Result<()> {
        self.w.begin_object()?;
        self.w.key("asset")?.begin_object()?;
        self.string("version", &doc.asset.version)?;
        self.opt_string("minVersion", doc.asset.min_version.as_ref())?;
        self.opt_string("generator", doc.asset.generator.as_ref())?;
        self.opt_string("copyright", doc.asset.copyright.as_ref())?;
        self.extra(&doc.asset.extra)?;
        self.w.end_object()?;
        self.strings("extensionsUsed", used)?;
        self.strings("extensionsRequired", &doc.extensions_required)?;
        self.opt_uint("scene", doc.scene)?;
        self.list("scenes", &doc.scenes, Self::scene)?;
        self.list("nodes", &doc.nodes, Self::node)?;
        self.list("meshes", &doc.meshes, Self::mesh)?;
        self.list("skins", &doc.skins, Self::skin)?;
        self.list("animations", &doc.animations, Self::animation)?;
        self.list("materials", &doc.materials, Self::material)?;
        self.list("textures", &doc.textures, Self::texture)?;
        self.list("images", &doc.images, Self::image)?;
        self.list("samplers", &doc.samplers, Self::sampler)?;
        self.list("cameras", &doc.cameras, Self::camera)?;
        self.list("accessors", &doc.accessors, Self::accessor)?;
        self.list("bufferViews", &doc.buffer_views, Self::buffer_view)?;
        self.list("buffers", &doc.buffers, Self::buffer)?;
        self.extra_with(&doc.extra, !doc.lights.is_empty(), |o| {
            o.begin_ext("KHR_lights_punctual")?;
            o.list("lights", &doc.lights, Self::light)?;
            o.w.end_object()?;
            Ok(())
        })?;
        self.w.end_object()?;
        Ok(())
    }

    fn scene(&mut self, s: &Scene) -> Result<()> {
        self.name(s.name.as_ref())?;
        self.indices("nodes", &s.nodes)?;
        self.extra(&s.extra)
    }

    fn node(&mut self, n: &Node) -> Result<()> {
        self.name(n.name.as_ref())?;
        self.indices("children", &n.children)?;
        self.opt_uint("camera", n.camera)?;
        self.opt_uint("mesh", n.mesh)?;
        self.opt_uint("skin", n.skin)?;
        match n.transform {
            Transform::Matrix(m) => self.floats("matrix", &m)?,
            Transform::Trs { translation, rotation, scale } => {
                self.floats_unless("translation", &translation, &[0.0; 3])?;
                self.floats_unless("rotation", &rotation, &[0.0, 0.0, 0.0, 1.0])?;
                self.floats_unless("scale", &scale, &[1.0; 3])?;
            }
        }
        if !n.weights.is_empty() {
            self.floats("weights", &n.weights)?;
        }
        self.extra_with(&n.extra, n.light.is_some(), |o| {
            o.begin_ext("KHR_lights_punctual")?;
            o.opt_uint("light", n.light)?;
            o.w.end_object()?;
            Ok(())
        })
    }

    fn attributes(&mut self, attributes: &[(String, usize)]) -> Result<()> {
        self.w.begin_object()?;
        for (name, index) in attributes {
            self.uint(name, *index)?;
        }
        self.w.end_object()?;
        Ok(())
    }

    fn mesh(&mut self, m: &Mesh) -> Result<()> {
        self.name(m.name.as_ref())?;
        self.w.key("primitives")?.begin_array()?;
        for p in &m.primitives {
            self.w.begin_object()?;
            self.w.key("attributes")?;
            self.attributes(&p.attributes)?;
            self.opt_uint("indices", p.indices)?;
            self.opt_uint("material", p.material)?;
            self.uint_unless("mode", p.mode.code() as usize, 4)?;
            if !p.targets.is_empty() {
                self.w.key("targets")?.begin_array()?;
                for t in &p.targets {
                    self.attributes(t)?;
                }
                self.w.end_array()?;
            }
            self.extra(&p.extra)?;
            self.w.end_object()?;
        }
        self.w.end_array()?;
        if !m.weights.is_empty() {
            self.floats("weights", &m.weights)?;
        }
        self.extra(&m.extra)
    }

    fn skin(&mut self, s: &Skin) -> Result<()> {
        self.name(s.name.as_ref())?;
        self.opt_uint("inverseBindMatrices", s.inverse_bind_matrices)?;
        self.opt_uint("skeleton", s.skeleton)?;
        self.indices("joints", &s.joints)?;
        self.extra(&s.extra)
    }

    fn animation(&mut self, a: &Animation) -> Result<()> {
        self.name(a.name.as_ref())?;
        self.list("channels", &a.channels, |o, c| {
            o.uint("sampler", c.sampler)?;
            o.w.key("target")?.begin_object()?;
            o.opt_uint("node", c.node)?;
            o.string("path", c.path.name())?;
            o.extra(&c.target_extra)?;
            o.w.end_object()?;
            o.extra(&c.extra)
        })?;
        self.list("samplers", &a.samplers, |o, s| {
            o.uint("input", s.input)?;
            o.uint("output", s.output)?;
            match s.interpolation {
                Interpolation::Linear => {}
                Interpolation::Step => o.string("interpolation", "STEP")?,
                Interpolation::CubicSpline => o.string("interpolation", "CUBICSPLINE")?,
            }
            o.extra(&s.extra)
        })?;
        self.extra(&a.extra)
    }

    /// A texture info; `scale_key` names its `scale` or `strength` member.
    fn texture_info(&mut self, key: &str, t: Option<&TextureInfo>, scale_key: Option<&str>) -> Result<()> {
        let Some(t) = t else { return Ok(()) };
        self.w.key(key)?.begin_object()?;
        self.uint("index", t.index)?;
        self.uint_unless("texCoord", t.tex_coord as usize, 0)?;
        if let Some(k) = scale_key {
            self.float_unless(k, t.scale, 1.0)?;
        }
        self.extra_with(&t.extra, t.transform.is_some(), |o| {
            if let Some(x) = &t.transform {
                o.begin_ext("KHR_texture_transform")?;
                o.floats_unless("offset", &x.offset, &[0.0; 2])?;
                o.float_unless("rotation", x.rotation, 0.0)?;
                o.floats_unless("scale", &x.scale, &[1.0; 2])?;
                if let Some(tc) = x.tex_coord {
                    o.uint("texCoord", tc as usize)?;
                }
                o.w.end_object()?;
            }
            Ok(())
        })?;
        self.w.end_object()?;
        Ok(())
    }

    fn tex(&mut self, key: &str, t: Option<&TextureInfo>) -> Result<()> {
        self.texture_info(key, t, None)
    }

    fn material(&mut self, m: &Material) -> Result<()> {
        self.name(m.name.as_ref())?;
        let p = &m.pbr;
        self.w.key("pbrMetallicRoughness")?.begin_object()?;
        self.floats_unless("baseColorFactor", &p.base_color_factor, &[1.0; 4])?;
        self.tex("baseColorTexture", p.base_color_texture.as_ref())?;
        self.float_unless("metallicFactor", p.metallic_factor, 1.0)?;
        self.float_unless("roughnessFactor", p.roughness_factor, 1.0)?;
        self.tex("metallicRoughnessTexture", p.metallic_roughness_texture.as_ref())?;
        self.extra(&p.extra)?;
        self.w.end_object()?;
        self.texture_info("normalTexture", m.normal_texture.as_ref(), Some("scale"))?;
        self.texture_info("occlusionTexture", m.occlusion_texture.as_ref(), Some("strength"))?;
        self.tex("emissiveTexture", m.emissive_texture.as_ref())?;
        self.floats_unless("emissiveFactor", &m.emissive_factor, &[0.0; 3])?;
        match m.alpha_mode {
            AlphaMode::Opaque => {}
            AlphaMode::Mask => self.string("alphaMode", "MASK")?,
            AlphaMode::Blend => self.string("alphaMode", "BLEND")?,
        }
        self.float_unless("alphaCutoff", m.alpha_cutoff, 0.5)?;
        if m.double_sided {
            self.w.key("doubleSided")?.bool(true)?;
        }
        let has_typed = m.unlit
            || m.emissive_strength.is_some()
            || m.ior.is_some()
            || m.specular.is_some()
            || m.clearcoat.is_some()
            || m.sheen.is_some()
            || m.transmission.is_some()
            || m.volume.is_some()
            || m.diffuse_transmission.is_some();
        self.extra_with(&m.extra, has_typed, |o| o.material_extensions(m))
    }

    fn material_extensions(&mut self, m: &Material) -> Result<()> {
        if m.unlit {
            self.begin_ext("KHR_materials_unlit")?;
            self.w.end_object()?;
        }
        if let Some(s) = m.emissive_strength {
            self.begin_ext("KHR_materials_emissive_strength")?;
            self.float_unless("emissiveStrength", s, 1.0)?;
            self.w.end_object()?;
        }
        if let Some(ior) = m.ior {
            self.begin_ext("KHR_materials_ior")?;
            self.float_unless("ior", ior, 1.5)?;
            self.w.end_object()?;
        }
        if let Some(s) = &m.specular {
            self.begin_ext("KHR_materials_specular")?;
            self.float_unless("specularFactor", s.factor, 1.0)?;
            self.tex("specularTexture", s.texture.as_ref())?;
            self.floats_unless("specularColorFactor", &s.color_factor, &[1.0; 3])?;
            self.tex("specularColorTexture", s.color_texture.as_ref())?;
            self.w.end_object()?;
        }
        if let Some(c) = &m.clearcoat {
            self.begin_ext("KHR_materials_clearcoat")?;
            self.float_unless("clearcoatFactor", c.factor, 0.0)?;
            self.tex("clearcoatTexture", c.texture.as_ref())?;
            self.float_unless("clearcoatRoughnessFactor", c.roughness_factor, 0.0)?;
            self.tex("clearcoatRoughnessTexture", c.roughness_texture.as_ref())?;
            self.texture_info("clearcoatNormalTexture", c.normal_texture.as_ref(), Some("scale"))?;
            self.w.end_object()?;
        }
        if let Some(s) = &m.sheen {
            self.begin_ext("KHR_materials_sheen")?;
            self.floats_unless("sheenColorFactor", &s.color_factor, &[0.0; 3])?;
            self.tex("sheenColorTexture", s.color_texture.as_ref())?;
            self.float_unless("sheenRoughnessFactor", s.roughness_factor, 0.0)?;
            self.tex("sheenRoughnessTexture", s.roughness_texture.as_ref())?;
            self.w.end_object()?;
        }
        if let Some(t) = &m.transmission {
            self.begin_ext("KHR_materials_transmission")?;
            self.float_unless("transmissionFactor", t.factor, 0.0)?;
            self.tex("transmissionTexture", t.texture.as_ref())?;
            self.w.end_object()?;
        }
        if let Some(v) = &m.volume {
            self.begin_ext("KHR_materials_volume")?;
            self.float_unless("thicknessFactor", v.thickness_factor, 0.0)?;
            self.tex("thicknessTexture", v.thickness_texture.as_ref())?;
            if let Some(d) = v.attenuation_distance {
                self.float("attenuationDistance", d)?;
            }
            self.floats_unless("attenuationColor", &v.attenuation_color, &[1.0; 3])?;
            self.w.end_object()?;
        }
        if let Some(d) = &m.diffuse_transmission {
            self.begin_ext("KHR_materials_diffuse_transmission")?;
            self.float_unless("diffuseTransmissionFactor", d.factor, 0.0)?;
            self.tex("diffuseTransmissionTexture", d.texture.as_ref())?;
            self.floats_unless("diffuseTransmissionColorFactor", &d.color_factor, &[1.0; 3])?;
            self.tex("diffuseTransmissionColorTexture", d.color_texture.as_ref())?;
            self.w.end_object()?;
        }
        Ok(())
    }

    fn texture(&mut self, t: &Texture) -> Result<()> {
        self.name(t.name.as_ref())?;
        self.opt_uint("sampler", t.sampler)?;
        self.opt_uint("source", t.source)?;
        self.extra_with(&t.extra, t.basisu_source.is_some(), |o| {
            o.begin_ext("KHR_texture_basisu")?;
            o.opt_uint("source", t.basisu_source)?;
            o.w.end_object()?;
            Ok(())
        })
    }

    fn image(&mut self, i: &Image) -> Result<()> {
        self.name(i.name.as_ref())?;
        self.opt_string("uri", i.uri.as_ref())?;
        self.opt_string("mimeType", i.mime_type.as_ref())?;
        self.opt_uint("bufferView", i.buffer_view)?;
        self.extra(&i.extra)
    }

    fn sampler(&mut self, s: &Sampler) -> Result<()> {
        self.name(s.name.as_ref())?;
        if let Some(f) = s.mag_filter {
            self.uint("magFilter", f as usize)?;
        }
        if let Some(f) = s.min_filter {
            self.uint("minFilter", f as usize)?;
        }
        self.uint_unless("wrapS", s.wrap_s as usize, REPEAT as usize)?;
        self.uint_unless("wrapT", s.wrap_t as usize, REPEAT as usize)?;
        self.extra(&s.extra)
    }

    fn camera(&mut self, c: &Camera) -> Result<()> {
        self.name(c.name.as_ref())?;
        match c.projection {
            Projection::Perspective { aspect_ratio, yfov, zfar, znear } => {
                self.string("type", "perspective")?;
                self.w.key("perspective")?.begin_object()?;
                if let Some(a) = aspect_ratio {
                    self.float("aspectRatio", a)?;
                }
                self.float("yfov", yfov)?;
                if let Some(f) = zfar {
                    self.float("zfar", f)?;
                }
                self.float("znear", znear)?;
            }
            Projection::Orthographic { xmag, ymag, zfar, znear } => {
                self.string("type", "orthographic")?;
                self.w.key("orthographic")?.begin_object()?;
                self.float("xmag", xmag)?;
                self.float("ymag", ymag)?;
                self.float("zfar", zfar)?;
                self.float("znear", znear)?;
            }
        }
        self.w.end_object()?;
        self.extra(&c.extra)
    }

    fn light(&mut self, l: &Light) -> Result<()> {
        self.name(l.name.as_ref())?;
        self.floats_unless("color", &l.color, &[1.0; 3])?;
        self.float_unless("intensity", l.intensity, 1.0)?;
        match l.kind {
            LightKind::Directional => self.string("type", "directional")?,
            LightKind::Point => self.string("type", "point")?,
            LightKind::Spot { inner_cone_angle, outer_cone_angle } => {
                self.string("type", "spot")?;
                self.w.key("spot")?.begin_object()?;
                self.float_unless("innerConeAngle", inner_cone_angle, 0.0)?;
                self.float_unless("outerConeAngle", outer_cone_angle, std::f32::consts::FRAC_PI_4)?;
                self.w.end_object()?;
            }
        }
        if let Some(r) = l.range {
            self.float("range", r)?;
        }
        self.extra(&l.extra)
    }

    /// `min` or `max`: floats for float accessors (shortest `f32` digits), integers otherwise.
    fn bounds(&mut self, key: &str, values: &[f64], component: ComponentType) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        self.w.key(key)?.begin_array_inline()?;
        for &v in values {
            if component == ComponentType::F32 {
                self.w.f32(v as f32)?;
            } else if v.fract() == 0.0 && v.abs() < 9.0e15 {
                self.w.i64(v as i64)?;
            } else {
                self.w.f64(v)?;
            }
        }
        self.w.end_array()?;
        Ok(())
    }

    fn accessor(&mut self, a: &Accessor) -> Result<()> {
        self.name(a.name.as_ref())?;
        self.opt_uint("bufferView", a.buffer_view)?;
        self.uint_unless("byteOffset", a.byte_offset, 0)?;
        self.uint("componentType", a.component_type.code() as usize)?;
        if a.normalized {
            self.w.key("normalized")?.bool(true)?;
        }
        self.uint("count", a.count)?;
        self.string("type", a.kind.name())?;
        self.bounds("min", &a.min, a.component_type)?;
        self.bounds("max", &a.max, a.component_type)?;
        if let Some(s) = &a.sparse {
            self.w.key("sparse")?.begin_object()?;
            self.uint("count", s.count)?;
            self.w.key("indices")?.begin_object()?;
            self.uint("bufferView", s.indices_view)?;
            self.uint_unless("byteOffset", s.indices_offset, 0)?;
            self.uint("componentType", s.indices_type.code() as usize)?;
            self.w.end_object()?;
            self.w.key("values")?.begin_object()?;
            self.uint("bufferView", s.values_view)?;
            self.uint_unless("byteOffset", s.values_offset, 0)?;
            self.w.end_object()?;
            self.extra(&s.extra)?;
            self.w.end_object()?;
        }
        self.extra(&a.extra)
    }

    fn buffer_view(&mut self, v: &BufferView) -> Result<()> {
        self.name(v.name.as_ref())?;
        self.uint("buffer", v.buffer)?;
        self.uint_unless("byteOffset", v.byte_offset, 0)?;
        self.uint("byteLength", v.byte_length)?;
        self.opt_uint("byteStride", v.byte_stride)?;
        if let Some(t) = v.target {
            self.uint("target", t.code() as usize)?;
        }
        self.extra(&v.extra)
    }

    fn buffer(&mut self, b: &Buffer) -> Result<()> {
        self.name(b.name.as_ref())?;
        self.opt_string("uri", b.uri.as_ref())?;
        self.uint("byteLength", b.byte_length)?;
        self.extra(&b.extra)
    }
}
