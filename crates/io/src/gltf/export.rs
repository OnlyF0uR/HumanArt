//! The export builder: packs typed arrays into one binary buffer and writes accessors with the
//! `min` / `max` the specification asks for, sparse morph targets and embedded images `[IO-10]`
//! `[EXP-P1]`.
//!
//! Every buffer view starts on a 4-byte boundary and every accessor is tightly packed, so the output
//! meets the alignment rules of the specification. [`Builder::finish`] validates the document.

use super::Gltf;
use super::doc::{Accessor, Buffer, BufferView, ComponentType, Document, ElementType, Extra, Image, Sparse, Target};
use super::validate::validate;
use super::write::used_extensions;
use crate::error::{Error, Format, Result};
use crate::image::Limits;

/// An [`Error::Encode`] for glTF.
const fn encode(what: &'static str) -> Error {
    Error::Encode { format: Format::Gltf, what }
}

/// Builds a [`Gltf`] whose data lives in one buffer (the GLB BIN chunk, or a `.bin` file or data URI
/// once a URI is set).
#[derive(Clone, Debug)]
pub struct Builder {
    /// The document under construction: add nodes, meshes, materials and so on here directly; the
    /// builder's methods add accessors, buffer views and images.
    pub document: Document,
    /// The buffer data.
    bin: Vec<u8>,
}

impl Builder {
    /// An empty glTF 2.0 document whose `asset.generator` is `generator`.
    #[must_use]
    pub fn new(generator: &str) -> Self {
        let mut document = Document::default();
        document.asset.generator = Some(generator.to_owned());
        Self { document, bin: Vec::new() }
    }

    /// The bytes added so far.
    #[must_use]
    pub fn bin_len(&self) -> usize {
        self.bin.len()
    }

    /// Appends `bytes` as a new buffer view starting on a 4-byte boundary and returns its index.
    pub fn view(&mut self, bytes: &[u8], byte_stride: Option<usize>, target: Option<Target>) -> Result<usize> {
        if bytes.is_empty() {
            return Err(encode("buffer view with no bytes"));
        }
        self.bin.resize(self.bin.len().next_multiple_of(4), 0);
        let byte_offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.document.buffer_views.push(BufferView {
            buffer: 0,
            byte_offset,
            byte_length: bytes.len(),
            byte_stride,
            target,
            name: None,
            extra: Extra::default(),
        });
        Ok(self.document.buffer_views.len() - 1)
    }

    fn push_accessor(&mut self, accessor: Accessor) -> usize {
        self.document.accessors.push(accessor);
        self.document.accessors.len() - 1
    }

    /// The element count of `len` components of `kind`, failing on an empty or ragged array.
    fn count(len: usize, kind: ElementType) -> Result<usize> {
        if len == 0 || !len.is_multiple_of(kind.components()) {
            return Err(encode("accessor data is empty or not a whole number of elements"));
        }
        Ok(len / kind.components())
    }

    /// Per-component `min` and `max` of float data, failing on NaN or infinity.
    fn bounds(data: &[f32], kind: ElementType) -> Result<(Vec<f64>, Vec<f64>)> {
        let comps = kind.components();
        let mut min = vec![f64::INFINITY; comps];
        let mut max = vec![f64::NEG_INFINITY; comps];
        for element in data.chunks_exact(comps) {
            for ((&v, lo), hi) in element.iter().zip(&mut min).zip(&mut max) {
                if !v.is_finite() {
                    return Err(encode("accessor data holds NaN or infinity"));
                }
                *lo = lo.min(f64::from(v));
                *hi = hi.max(f64::from(v));
            }
        }
        Ok((min, max))
    }

    /// A float accessor of `kind` over `data` (`count × components` values), with `min` and `max`.
    /// `target` is [`Target::ArrayBuffer`] for vertex attributes and `None` for other data
    /// (inverse bind matrices, animation keyframes).
    pub fn accessor_f32(&mut self, data: &[f32], kind: ElementType, target: Option<Target>) -> Result<usize> {
        if matches!(kind, ElementType::Mat2 | ElementType::Mat3) {
            return Err(encode("MAT2 and MAT3 float accessors are written as columns of MAT4 or vectors"));
        }
        let count = Self::count(data.len(), kind)?;
        let (min, max) = Self::bounds(data, kind)?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, None, target)?;
        Ok(self.push_accessor(Accessor {
            buffer_view: Some(view),
            byte_offset: 0,
            component_type: ComponentType::F32,
            normalized: false,
            count,
            kind,
            min,
            max,
            sparse: None,
            name: None,
            extra: Extra::default(),
        }))
    }

    /// An index accessor: `u16` when every index is below 65535 (the primitive restart value is
    /// never written), `u32` otherwise.
    pub fn accessor_indices(&mut self, indices: &[u32]) -> Result<usize> {
        if indices.is_empty() {
            return Err(encode("index accessor with no indices"));
        }
        let (lo, hi) = indices.iter().fold((u32::MAX, 0), |(lo, hi), &i| (lo.min(i), hi.max(i)));
        let (component_type, bytes): (_, Vec<u8>) = if hi < 65535 {
            (ComponentType::U16, indices.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect())
        } else {
            (ComponentType::U32, indices.iter().flat_map(|&i| i.to_le_bytes()).collect())
        };
        let view = self.view(&bytes, None, Some(Target::ElementArrayBuffer))?;
        Ok(self.push_accessor(Accessor {
            buffer_view: Some(view),
            byte_offset: 0,
            component_type,
            normalized: false,
            count: indices.len(),
            kind: ElementType::Scalar,
            min: vec![f64::from(lo)],
            max: vec![f64::from(hi)],
            sparse: None,
            name: None,
            extra: Extra::default(),
        }))
    }

    /// A `JOINTS_n` accessor: `u8` components when every joint index is below 256, `u16` otherwise.
    pub fn accessor_joints(&mut self, joints: &[[u16; 4]]) -> Result<usize> {
        if joints.is_empty() {
            return Err(encode("joints accessor with no vertices"));
        }
        let small = joints.iter().flatten().all(|&j| j < 256);
        let (component_type, bytes): (_, Vec<u8>) = if small {
            (ComponentType::U8, joints.iter().flatten().map(|&j| j as u8).collect())
        } else {
            (ComponentType::U16, joints.iter().flatten().flat_map(|j| j.to_le_bytes()).collect())
        };
        let view = self.view(&bytes, None, Some(Target::ArrayBuffer))?;
        Ok(self.push_accessor(Accessor {
            buffer_view: Some(view),
            byte_offset: 0,
            component_type,
            normalized: false,
            count: joints.len(),
            kind: ElementType::Vec4,
            min: Vec::new(),
            max: Vec::new(),
            sparse: None,
            name: None,
            extra: Extra::default(),
        }))
    }

    /// A morph target accessor of `kind` over the dense `deltas` (`count × components`).
    ///
    /// Only the elements with a non-zero component are stored: as a sparse accessor without a
    /// buffer view when that is smaller than the dense array (the usual case for a facial target,
    /// which moves a small part of the mesh), densely otherwise, and as an accessor with neither
    /// buffer view nor sparse data when every delta is zero. `min` and `max` cover every element,
    /// zeros included, as the specification requires for `POSITION` targets.
    pub fn accessor_morph(&mut self, deltas: &[f32], kind: ElementType) -> Result<usize> {
        if !matches!(kind, ElementType::Vec2 | ElementType::Vec3 | ElementType::Vec4) {
            return Err(encode("morph target accessors are VEC2, VEC3 or VEC4"));
        }
        let count = Self::count(deltas.len(), kind)?;
        let (min, max) = Self::bounds(deltas, kind)?;
        let comps = kind.components();
        let moved: Vec<usize> = deltas
            .chunks_exact(comps)
            .enumerate()
            .filter(|(_, e)| e.iter().any(|&v| v != 0.0))
            .map(|(i, _)| i)
            .collect();
        let elem = comps * 4;
        let index_type = match count {
            0..=256 => ComponentType::U8,
            257..=65536 => ComponentType::U16,
            _ => ComponentType::U32,
        };
        let sparse_size = moved.len() * (index_type.size() + elem);
        if !moved.is_empty() && sparse_size >= count * elem {
            return self.accessor_f32(deltas, kind, Some(Target::ArrayBuffer));
        }
        let sparse = if moved.is_empty() {
            None
        } else {
            let indices: Vec<u8> = match index_type {
                ComponentType::U8 => moved.iter().map(|&i| i as u8).collect(),
                ComponentType::U16 => moved.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(),
                _ => moved.iter().flat_map(|&i| (i as u32).to_le_bytes()).collect(),
            };
            let values: Vec<u8> = moved
                .iter()
                .flat_map(|&i| deltas.get(i * comps..(i + 1) * comps).unwrap_or(&[]))
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let indices_view = self.view(&indices, None, None)?;
            let values_view = self.view(&values, None, None)?;
            Some(Sparse {
                count: moved.len(),
                indices_view,
                indices_offset: 0,
                indices_type: index_type,
                values_view,
                values_offset: 0,
                extra: Extra::default(),
            })
        };
        Ok(self.push_accessor(Accessor {
            buffer_view: None,
            byte_offset: 0,
            component_type: ComponentType::F32,
            normalized: false,
            count,
            kind,
            min,
            max,
            sparse,
            name: None,
            extra: Extra::default(),
        }))
    }

    /// Embeds an encoded image (`image/png`, `image/jpeg`, `image/ktx2`) in the buffer and returns
    /// its image index.
    pub fn image(&mut self, bytes: &[u8], mime_type: &str, name: Option<&str>) -> Result<usize> {
        let view = self.view(bytes, None, None)?;
        self.document.images.push(Image {
            uri: None,
            mime_type: Some(mime_type.to_owned()),
            buffer_view: Some(view),
            name: name.map(str::to_owned),
            extra: Extra::default(),
        });
        Ok(self.document.images.len() - 1)
    }

    /// Finishes the document: one buffer without a URI holding the data (none when nothing was
    /// added; buffers set on [`Builder::document`] are replaced), validation against
    /// [`Limits::DEFAULT`], and `extensionsUsed` filled in with [`super::used_extensions`].
    pub fn finish(self) -> Result<Gltf> {
        let Self { mut document, mut bin } = self;
        let buffers = if bin.is_empty() {
            Vec::new()
        } else {
            bin.resize(bin.len().next_multiple_of(4), 0);
            document.buffers = vec![Buffer { uri: None, byte_length: bin.len(), name: None, extra: Extra::default() }];
            vec![bin]
        };
        validate(&document, &buffers, &Limits::DEFAULT)?;
        document.extensions_used = used_extensions(&document)?;
        Ok(Gltf { document, buffers })
    }
}
