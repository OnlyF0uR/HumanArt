//! The character model: template topology, regions, UVs, identity, expression, skeleton and skinning data, and the CPU reference evaluation.
//!
//! Module map (WP-06, first half: the template data model):
//!
//! - [`error`]: [`ModelError`], the crate's error enum.
//! - [`names`]: stable-name rules shared by every table, and [`Side`].
//! - [`mesh`]: [`Mesh`] (SoA positions and UVs, quad-dominant polygon faces, weld and normal keys for
//!   UV seams and hard edges, parts), [`MeshBuilder`], triangulation `[MOD-01]`.
//! - [`topology`]: edges and CSR adjacency derived from a mesh `[EVAL-04]`.
//! - [`region`]: the region hierarchy, per-vertex regions and soft weights `[MOD-03]`.
//! - [`landmark`]: barycentric landmarks `[MOD-06]`.
//! - [`skeleton`]: joints, rest transforms, limits and the joint regressor `[MOD-07]`.
//! - [`skin`]: fixed-count normalised skin weights and the dual-quaternion region map `[SKIN-01]`.
//! - [`delta`]: sparse f16 delta bases sorted by region `[EVAL-02]`.
//! - [`expression`]: expression targets, combination and pose-space correctives `[EXP-01]`
//!   `[EXP-02]` `[SKIN-02]`.
//! - [`identity`]: the authored identity axes, semantic and asymmetry components with their
//!   joint-regressor rows (no-data operation) `[ID-01]` `[ID-03]` `[ID-05]` `[ID-07]`.
//! - [`lod`]:LOD1..LOD4 with barycentric sources for attribute transfer, LOD selection `[MOD-05]`.
//! - [`simplify`]: the attribute-preserving LOD generator (constrained quadric edge collapse,
//!   per-level transfer, error report) and the LOD switching metric `[MOD-05]` `[AC-WP28-1]`.
//! - [`template`]: [`Template`], the checked whole, with its UV tile table `[MOD-04]` and info.
//! - [`package`]: the `.antmpl` container: [`write_package`], [`read_package`] `[IO-21]` `[IO-24]`.
//! - [`validate`]: the template validation report (watertight, manifold, region coverage, UV
//!   overlap, texel density, landmarks, weights, deltas, LOD error) `[AC-WP06-1]`.
#![forbid(unsafe_code)]

pub mod delta;
pub mod error;
pub mod eval;
pub mod expression;
pub mod eye;
pub mod generate;
pub mod hair;
pub mod identity;
pub mod landmark;
pub mod lod;
pub mod mesh;
pub mod mouth;
pub mod names;
pub mod package;
pub mod region;
pub mod scanfit;
pub mod simplify;
pub mod skeleton;
pub mod skin;
pub mod skin_appearance;
pub mod template;
pub mod topology;
pub mod validate;

pub use error::{ModelError, Result};
pub use mesh::{Mesh, MeshBuilder, MeshData, MeshPart, Triangulation};
pub use names::Side;
pub use package::{PackageLimits, read_package, write_package};
pub use template::{Template, TemplateInfo, TemplateParts, UvTile};
pub use topology::{Csr, Topology};
