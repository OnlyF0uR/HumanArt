//! Sparse delta bases `[EVAL-02]`: every identity axis, semantic slider, expression target and
//! corrective of the template as a sparse list of vertex displacements.
//!
//! Component `c` stores the vertex indices it moves (strictly increasing) and one f16 xyz
//! displacement per index, in metres per unit weight. The components are **sorted by their
//! bounding region** (stable within a region), so a batch of components touches a coherent vertex
//! set on the GPU. All components share three flat arrays (CSR layout), which upload to storage
//! buffers as they are.
//!
//! The bases hold whatever the template has: with no scan data (`[ID-07]`) there are simply no
//! identity components and only authored semantic, expression and corrective deltas; a fitted model
//! later fills the same arrays.

use humanart_math::{F16, Vec3};

use crate::error::{ModelError, Result};
use crate::names::{check_name, check_names};

/// The most delta components a template may have.
pub const MAX_COMPONENTS: usize = 1 << 16;
/// The most delta entries (index + xyz) over all components.
pub const MAX_DELTA_ENTRIES: usize = 1 << 28;

/// What a delta component is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeltaKind {
    /// A global identity component (PCA over the whole body or head + neck) `[ID-01]` (a).
    IdentityGlobal,
    /// A region-local identity component `[ID-01]` (b).
    IdentityRegional,
    /// A named semantic axis, authored or regressed `[ID-01]` (c), `[ID-03]`.
    Semantic,
    /// An asymmetry component `[ID-05]`.
    Asymmetry,
    /// An expression target `[EXP-01]`.
    Expression,
    /// A combination corrective `[EXP-02]`.
    CombinationCorrective,
    /// A pose-space corrective `[SKIN-02]`.
    PoseCorrective,
}

impl DeltaKind {
    /// Every kind, in stored order.
    pub const ALL: [Self; 7] = [
        Self::IdentityGlobal,
        Self::IdentityRegional,
        Self::Semantic,
        Self::Asymmetry,
        Self::Expression,
        Self::CombinationCorrective,
        Self::PoseCorrective,
    ];

    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// The kind for a stored byte, if valid.
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(v as usize).copied()
    }

    /// Whether this kind comes from scan statistics (absent in no-data operation `[ID-07]`).
    #[must_use]
    pub const fn is_scan_derived(self) -> bool {
        matches!(self, Self::IdentityGlobal | Self::IdentityRegional | Self::Asymmetry)
    }
}

/// The description of one component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaComponent {
    /// Stable name.
    pub name: String,
    /// What it is.
    pub kind: DeltaKind,
    /// The bounding region: every vertex the component moves lies in this region's subtree.
    pub region: u16,
}

/// The delta bases of a template. Build with [`DeltaBasesBuilder`] or [`DeltaBases::from_parts`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeltaBases {
    components: Vec<DeltaComponent>,
    offsets: Vec<u32>,
    indices: Vec<u32>,
    deltas: Vec<[F16; 3]>,
}

impl DeltaBases {
    /// Bases with no components.
    #[must_use]
    pub fn empty() -> Self {
        Self { offsets: vec![0], ..Self::default() }
    }

    /// Checks raw arrays and wraps them: `offsets` has `components.len() + 1` entries from 0 and
    /// ends at the entry count, indices strictly increase within a component, deltas are finite,
    /// components are sorted by region and names are valid and unique. Vertex ranges are checked
    /// against a mesh by [`check_vertices`](Self::check_vertices).
    pub fn from_parts(
        components: Vec<DeltaComponent>,
        offsets: Vec<u32>,
        indices: Vec<u32>,
        deltas: Vec<[F16; 3]>,
    ) -> Result<Self> {
        if components.len() > MAX_COMPONENTS {
            return Err(ModelError::limit("delta component count", MAX_COMPONENTS as u64, components.len() as u64));
        }
        if indices.len() > MAX_DELTA_ENTRIES {
            return Err(ModelError::limit("delta entry count", MAX_DELTA_ENTRIES as u64, indices.len() as u64));
        }
        check_names("delta component", components.iter().map(|c| c.name.as_str()))?;
        if offsets.len() != components.len() + 1 || offsets.first() != Some(&0) {
            return Err(ModelError::invalid(
                "delta bases",
                0,
                "offsets do not have component_count + 1 entries from 0",
            ));
        }
        if indices.len() != deltas.len() || offsets.last().copied() != Some(indices.len() as u32) {
            return Err(ModelError::invalid("delta bases", 0, "entry arrays differ from the offsets"));
        }
        for (c, w) in offsets.windows(2).enumerate() {
            let Some(own) = indices.get(w[0] as usize..w[1] as usize) else {
                return Err(ModelError::invalid("delta component", c, "offsets decrease or overrun"));
            };
            if own.windows(2).any(|p| p[0] >= p[1]) {
                return Err(ModelError::invalid("delta component", c, "vertex indices not strictly increasing"));
            }
        }
        if let Some(i) = deltas.iter().position(|d| !d.iter().all(|x| x.is_finite())) {
            return Err(ModelError::invalid("delta entry", i, "displacement is not finite"));
        }
        if let Some(c) = components.windows(2).position(|w| w[0].region > w[1].region) {
            return Err(ModelError::invalid("delta component", c + 1, "components are not sorted by region"));
        }
        Ok(Self { components, offsets, indices, deltas })
    }

    /// Checks that every index is below `vertex_count` and every region below `region_count`.
    pub fn check_vertices(&self, vertex_count: usize, region_count: usize) -> Result<()> {
        for (c, comp) in self.components.iter().enumerate() {
            if comp.region as usize >= region_count {
                return Err(ModelError::invalid("delta component", c, "region index out of range"));
            }
            // Indices increase, so the last one is the largest.
            if self.indices(c).last().is_some_and(|&v| v as usize >= vertex_count) {
                return Err(ModelError::invalid("delta component", c, "vertex index out of range"));
            }
        }
        Ok(())
    }

    /// The raw arrays back: components, offsets, indices, deltas.
    #[must_use]
    pub fn into_parts(self) -> (Vec<DeltaComponent>, Vec<u32>, Vec<u32>, Vec<[F16; 3]>) {
        (self.components, self.offsets, self.indices, self.deltas)
    }

    /// The component descriptions.
    #[must_use]
    pub fn components(&self) -> &[DeltaComponent] {
        &self.components
    }

    /// How many components there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether there are no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// The `len() + 1` offsets.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// All vertex indices, component after component.
    #[must_use]
    pub fn all_indices(&self) -> &[u32] {
        &self.indices
    }

    /// All displacements, aligned with [`all_indices`](Self::all_indices).
    #[must_use]
    pub fn all_deltas(&self) -> &[[F16; 3]] {
        &self.deltas
    }

    /// The vertex indices of component `c` (empty when out of range).
    #[must_use]
    pub fn indices(&self, c: usize) -> &[u32] {
        self.range(c).map_or(&[], |r| &self.indices[r])
    }

    /// The displacements of component `c` (empty when out of range).
    #[must_use]
    pub fn deltas(&self, c: usize) -> &[[F16; 3]] {
        self.range(c).map_or(&[], |r| &self.deltas[r])
    }

    fn range(&self, c: usize) -> Option<std::ops::Range<usize>> {
        Some(*self.offsets.get(c)? as usize..*self.offsets.get(c + 1)? as usize)
    }

    /// The index of the component called `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<u32> {
        self.components.iter().position(|c| c.name == name).map(|i| i as u32)
    }

    /// How many components of kind `kind` there are.
    #[must_use]
    pub fn count_kind(&self, kind: DeltaKind) -> usize {
        self.components.iter().filter(|c| c.kind == kind).count()
    }

    /// Converts one stored displacement to f32.
    #[must_use]
    pub fn delta_f32(d: [F16; 3]) -> Vec3 {
        Vec3::new(d[0].to_f32(), d[1].to_f32(), d[2].to_f32())
    }

    /// Adds `weight` times component `c` to `positions` (the CPU form of one accumulation step of
    /// `[EVAL-02]`). Fails without changing anything when `c` does not exist or an index is outside
    /// `positions`.
    pub fn accumulate(&self, c: usize, weight: f32, positions: &mut [Vec3]) -> Result<()> {
        let range = self.range(c).ok_or(ModelError::invalid("delta component", c, "component does not exist"))?;
        let indices = &self.indices[range.clone()];
        if indices.last().is_some_and(|&v| v as usize >= positions.len()) {
            return Err(ModelError::invalid("delta component", c, "vertex index out of range"));
        }
        for (&v, &d) in indices.iter().zip(&self.deltas[range]) {
            positions[v as usize] += Self::delta_f32(d) * weight;
        }
        Ok(())
    }
}

/// Collects components in any order and produces [`DeltaBases`] sorted by region.
#[derive(Clone, Debug, Default)]
pub struct DeltaBasesBuilder {
    components: Vec<(DeltaComponent, Vec<u32>, Vec<[F16; 3]>)>,
}

impl DeltaBasesBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a component from `(vertex, displacement)` pairs in any order. Displacements are rounded
    /// to f16; entries that round to zero are dropped (sparsity), a vertex given twice is an error.
    pub fn add(&mut self, name: &str, kind: DeltaKind, region: u16, entries: &[(u32, Vec3)]) -> Result<()> {
        check_name("delta component", name)?;
        let mut sorted: Vec<(u32, [F16; 3])> = Vec::with_capacity(entries.len());
        for &(v, d) in entries {
            if !d.is_finite() {
                return Err(ModelError::invalid("delta entry", v as usize, "displacement is not finite"));
            }
            let h = [F16::from_f32(d.x), F16::from_f32(d.y), F16::from_f32(d.z)];
            if !h.iter().all(|x| x.is_finite()) {
                return Err(ModelError::invalid("delta entry", v as usize, "displacement exceeds the f16 range"));
            }
            if h.iter().any(|x| x.to_f32() != 0.0) {
                sorted.push((v, h));
            }
        }
        sorted.sort_unstable_by_key(|e| e.0);
        if let Some(w) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(ModelError::invalid("delta entry", w[0].0 as usize, "vertex given twice"));
        }
        let (indices, deltas) = sorted.into_iter().unzip();
        self.components.push((DeltaComponent { name: name.to_owned(), kind, region }, indices, deltas));
        Ok(())
    }

    /// Sorts the components by region (stable) and checks the result.
    pub fn finish(mut self) -> Result<DeltaBases> {
        self.components.sort_by_key(|c| c.0.region);
        let mut components = Vec::with_capacity(self.components.len());
        let mut offsets = vec![0_u32];
        let (mut indices, mut deltas) = (Vec::new(), Vec::new());
        for (c, i, d) in self.components {
            components.push(c);
            indices.extend_from_slice(&i);
            deltas.extend_from_slice(&d);
            offsets.push(indices.len() as u32);
        }
        DeltaBases::from_parts(components, offsets, indices, deltas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DeltaBases {
        let mut b = DeltaBasesBuilder::new();
        b.add("jaw-open", DeltaKind::Expression, 3, &[(5, Vec3::new(0.0, -0.01, 0.0)), (2, Vec3::Y * 0.5)]).unwrap();
        b.add("nose-width", DeltaKind::Semantic, 1, &[(0, Vec3::X), (1, Vec3::ZERO)]).unwrap();
        b.add("mass", DeltaKind::Semantic, 0, &[]).unwrap();
        b.finish().unwrap()
    }

    #[test]
    fn builder_sorts_by_region_and_drops_zeros() {
        let d = sample();
        assert_eq!(d.len(), 3);
        let names: Vec<&str> = d.components().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["mass", "nose-width", "jaw-open"]);
        assert!(d.indices(0).is_empty());
        assert_eq!(d.indices(1), &[0]);
        assert_eq!(d.indices(2), &[2, 5]);
        assert_eq!(d.find("jaw-open"), Some(2));
        assert_eq!(d.count_kind(DeltaKind::Semantic), 2);
        assert_eq!(d.count_kind(DeltaKind::IdentityGlobal), 0);
        assert!(d.indices(3).is_empty() && d.deltas(3).is_empty());
        d.check_vertices(6, 4).unwrap();
        assert!(d.check_vertices(5, 4).is_err());
        assert!(d.check_vertices(6, 3).is_err());
        let (c, o, i, x) = d.clone().into_parts();
        assert_eq!(DeltaBases::from_parts(c, o, i, x).unwrap(), d);
    }

    #[test]
    fn accumulate_adds_f16_deltas() {
        let d = sample();
        let mut p = vec![Vec3::ZERO; 6];
        d.accumulate(2, 2.0, &mut p).unwrap();
        assert_eq!(p[2], Vec3::new(0.0, 1.0, 0.0));
        // -0.01 in f16 is within 2^-17 of the f32 value.
        assert!((p[5].y + 0.02).abs() < 2e-5, "{}", p[5].y);
        assert!(d.accumulate(3, 1.0, &mut p).is_err());
        let before = p.clone();
        assert!(d.accumulate(2, 1.0, &mut p[..5]).is_err());
        assert_eq!(p, before);
    }

    #[test]
    fn refuses_bad_components() {
        let mut b = DeltaBasesBuilder::new();
        assert!(b.add("x", DeltaKind::Semantic, 0, &[(1, Vec3::X), (1, Vec3::Y)]).is_err());
        assert!(b.add("x", DeltaKind::Semantic, 0, &[(1, Vec3::splat(f32::NAN))]).is_err());
        assert!(b.add("x", DeltaKind::Semantic, 0, &[(1, Vec3::splat(1e6))]).is_err());
        assert!(b.add("", DeltaKind::Semantic, 0, &[]).is_err());
        b.add("x", DeltaKind::Semantic, 0, &[]).unwrap();
        b.add("x", DeltaKind::Semantic, 0, &[]).unwrap();
        assert!(b.finish().is_err());

        let (c, o, i, x) = sample().into_parts();
        assert!(DeltaBases::from_parts(c.clone(), o[..3].to_vec(), i.clone(), x.clone()).is_err());
        let mut swapped = i.clone();
        swapped.swap(1, 2);
        assert!(DeltaBases::from_parts(c.clone(), o.clone(), swapped, x.clone()).is_err());
        let mut unsorted = c.clone();
        unsorted[0].region = 9;
        assert!(DeltaBases::from_parts(unsorted, o.clone(), i.clone(), x.clone()).is_err());
        let mut inf = x.clone();
        inf[0][1] = F16::INFINITY;
        assert!(DeltaBases::from_parts(c.clone(), o.clone(), i.clone(), inf).is_err());
        assert!(DeltaBases::from_parts(c, o, i, x[..2].to_vec()).is_err());
        assert_eq!(DeltaBases::empty().offsets(), &[0]);
        for k in DeltaKind::ALL {
            assert_eq!(DeltaKind::from_u8(k.to_u8()), Some(k));
        }
        assert_eq!(DeltaKind::from_u8(7), None);
        assert!(DeltaKind::IdentityGlobal.is_scan_derived() && !DeltaKind::Semantic.is_scan_derived());
    }
}
