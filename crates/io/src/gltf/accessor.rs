//! Reading accessor data: strides, matrix column padding, normalized integers and sparse
//! substitution `[IO-10]`.
//!
//! Every read checks each byte range again (with `get`), so these functions are safe on documents
//! that were never validated; on validated documents they only fail on a type mismatch.

use super::Gltf;
use super::doc::{Accessor, ComponentType, ElementType, Primitive};
use super::parse::invalid;
use super::validate::{sparse_index, view_bytes};
use crate::error::Result;

/// One component as `f32`, normalized integers mapped as the specification asks (`i8`:
/// `max(c / 127, -1)`, `u8`: `c / 255`, and the same for 16 bits).
fn component_f32(kind: ComponentType, normalized: bool, b: &[u8]) -> f32 {
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    match kind {
        ComponentType::I8 => {
            let v = f32::from(at(0) as i8);
            if normalized { (v / 127.0).max(-1.0) } else { v }
        }
        ComponentType::U8 => {
            let v = f32::from(at(0));
            if normalized { v / 255.0 } else { v }
        }
        ComponentType::I16 => {
            let v = f32::from(i16::from_le_bytes([at(0), at(1)]));
            if normalized { (v / 32767.0).max(-1.0) } else { v }
        }
        ComponentType::U16 => {
            let v = f32::from(u16::from_le_bytes([at(0), at(1)]));
            if normalized { v / 65535.0 } else { v }
        }
        ComponentType::U32 => u32::from_le_bytes([at(0), at(1), at(2), at(3)]) as f32,
        ComponentType::F32 => f32::from_le_bytes([at(0), at(1), at(2), at(3)]),
    }
}

/// One unsigned integer component as `u32` (the caller has checked the type).
fn component_u32(kind: ComponentType, _normalized: bool, b: &[u8]) -> u32 {
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    match kind.size() {
        1 => u32::from(at(0)),
        2 => u32::from(u16::from_le_bytes([at(0), at(1)])),
        _ => u32::from_le_bytes([at(0), at(1), at(2), at(3)]),
    }
}

/// Decodes the element at `base` of `bytes` into `out` (one slot per component, columns in order,
/// padding skipped).
fn element<T>(
    a: &Accessor,
    bytes: &[u8],
    base: usize,
    out: &mut [T],
    conv: fn(ComponentType, bool, &[u8]) -> T,
) -> Result<()> {
    let (cols, rows) = a.kind.shape();
    let size = a.component_type.size();
    let column = a.kind.column_stride(a.component_type);
    for col in 0..cols {
        for row in 0..rows {
            let at = base.saturating_add(col * column + row * size);
            let b = bytes
                .get(at..at.saturating_add(size))
                .ok_or(invalid("accessor reaches past the end of its bufferView"))?;
            if let Some(slot) = out.get_mut(col * rows + row) {
                *slot = conv(a.component_type, a.normalized, b);
            }
        }
    }
    Ok(())
}

impl Gltf {
    /// The accessor at `index`.
    pub fn accessor(&self, index: usize) -> Result<&Accessor> {
        self.document.accessors.get(index).ok_or(invalid("accessor index is out of range"))
    }

    /// Decodes accessor `index` into `count × components` values with `conv`.
    fn decode<T: Copy + Default>(&self, index: usize, conv: fn(ComponentType, bool, &[u8]) -> T) -> Result<Vec<T>> {
        let a = self.accessor(index)?;
        let doc = &self.document;
        let comps = a.kind.components();
        let elem = a.kind.size(a.component_type);
        let total = a.count.checked_mul(comps).ok_or(invalid("accessor.count overflows"))?;
        let mut out = vec![T::default(); total];
        if let Some(view) = a.buffer_view {
            let bytes = view_bytes(doc, &self.buffers, view)?;
            let stride = doc.buffer_views.get(view).and_then(|v| v.byte_stride).unwrap_or(elem);
            for (i, slot) in out.chunks_exact_mut(comps).enumerate() {
                element(a, bytes, a.byte_offset.saturating_add(i.saturating_mul(stride)), slot, conv)?;
            }
        }
        if let Some(sparse) = &a.sparse {
            let indices = view_bytes(doc, &self.buffers, sparse.indices_view)?
                .get(sparse.indices_offset..)
                .ok_or(invalid("sparse indices reach past their bufferView"))?;
            let values = view_bytes(doc, &self.buffers, sparse.values_view)?
                .get(sparse.values_offset..)
                .ok_or(invalid("sparse values reach past their bufferView"))?;
            for j in 0..sparse.count {
                let i = sparse_index(indices, sparse.indices_type, j)?;
                let start = i.checked_mul(comps).ok_or(invalid("sparse index is not below accessor.count"))?;
                let slot =
                    out.get_mut(start..start + comps).ok_or(invalid("sparse index is not below accessor.count"))?;
                element(a, values, j.saturating_mul(elem), slot, conv)?;
            }
        }
        Ok(out)
    }

    /// Accessor `index` as floats, `count × components` of them (a `MAT4` gives 16 per element in
    /// column order). Normalized integers map to `[0, 1]` or `[-1, 1]`, other integers convert as
    /// values; sparse substitution is applied.
    pub fn read_f32(&self, index: usize) -> Result<Vec<f32>> {
        self.decode(index, component_f32)
    }

    /// Accessor `index` as `N`-component float elements; fails unless the accessor has exactly `N`
    /// components per element.
    pub fn read_vec<const N: usize>(&self, index: usize) -> Result<Vec<[f32; N]>> {
        if self.accessor(index)?.kind.components() != N {
            return Err(invalid("accessor has a different number of components than asked for"));
        }
        let flat = self.read_f32(index)?;
        let (elements, _) = flat.as_chunks::<N>();
        Ok(elements.to_vec())
    }

    /// Accessor `index` as unsigned integers (`count × components`); fails unless its components are
    /// `u8`, `u16` or `u32` and not normalized (indices, joints).
    pub fn read_u32(&self, index: usize) -> Result<Vec<u32>> {
        let a = self.accessor(index)?;
        if a.normalized || !matches!(a.component_type, ComponentType::U8 | ComponentType::U16 | ComponentType::U32) {
            return Err(invalid("accessor is not an unsigned integer accessor"));
        }
        self.decode(index, component_u32)
    }

    /// The vertex indices of a primitive: its `indices` accessor, or `0..n` for a non-indexed
    /// primitive. Every index is checked against the vertex count.
    pub fn read_indices(&self, primitive: &Primitive) -> Result<Vec<u32>> {
        let vertices = self.vertex_count(primitive)?;
        if let Some(index) = primitive.indices {
            let indices = self.read_u32(index)?;
            if indices.iter().any(|&i| i as usize >= vertices) {
                return Err(invalid("vertex index is not below the vertex count"));
            }
            Ok(indices)
        } else {
            let n = u32::try_from(vertices).map_err(|_| invalid("vertex count does not fit u32"))?;
            Ok((0..n).collect())
        }
    }

    /// The vertex count of a primitive: the count of its `POSITION` accessor, or of its first
    /// attribute when it has none.
    pub fn vertex_count(&self, primitive: &Primitive) -> Result<usize> {
        let index = primitive
            .attribute("POSITION")
            .or_else(|| primitive.attributes.first().map(|&(_, a)| a))
            .ok_or(invalid("primitive has no attributes"))?;
        Ok(self.accessor(index)?.count)
    }

    /// The inverse bind matrices of skin `index`, one column-major matrix per joint (identity when
    /// the skin has none).
    pub fn inverse_bind_matrices(&self, index: usize) -> Result<Vec<[f32; 16]>> {
        let skin = self.document.skins.get(index).ok_or(invalid("skin index is out of range"))?;
        if let Some(a) = skin.inverse_bind_matrices {
            if self.accessor(a)?.kind != ElementType::Mat4 {
                return Err(invalid("skin.inverseBindMatrices is not a MAT4 accessor"));
            }
            let mut matrices = self.read_vec::<16>(a)?;
            matrices.truncate(skin.joints.len());
            Ok(matrices)
        } else {
            let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            Ok(vec![identity; skin.joints.len()])
        }
    }
}
