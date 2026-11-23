//! The authored identity model of no-data operation (WP-06, `[ID-01]` (c), `[ID-03]`, `[ID-05]`,
//! `[ID-07]`, `[MOD-07]`).
//!
//! With no scan data the identity model is the template plus **authored** components: the global
//! axes of `[ID-03]` (age, body mass, muscularity, height, sex-related shape), the global
//! asymmetry amount and the per-region asymmetry components of `[ID-05]`, and the named semantic
//! sliders of `[ID-01]` (c) (body proportions, hands, feet, face, ears, neck, eyeball, teeth). Each
//! is one sparse delta component of the template's [`DeltaBases`](crate::delta::DeltaBases), named
//! exactly as the parameter registry's `component` target (`global.height`, `semantic.nose-length`,
//! `asymmetry.eyes`), so the registry binds to it by name (`[REG-01]`, [`BINDINGS`]), plus a row of
//! the joint regressor so the skeleton follows the shape (`[MOD-07]`: height, limb lengths, hip and
//! shoulder width, neck length, eye position, hands and feet move their joints).
//!
//! **Kinds.** Every component here is [`DeltaKind::Semantic`](crate::delta::DeltaKind::Semantic),
//! also the asymmetry components: `DeltaKind::Asymmetry` (like the two identity kinds) is reserved
//! for components extracted from scans, so a template built here still reports zero scan-derived
//! components (`[ID-07]`, `[AC-WP06-3]`). The fitted model later adds its own components; the
//! registry keeps binding by name.
//!
//! **Measured where the axis claims a measurement, authored elsewhere.** Where an axis claims a
//! real-world measurement (stature, weight, shoulder and hip breadth, waist, thigh, buttock depth,
//! limb, hand and foot lengths, interpupillary distance, ear length, neck circumference), its amount
//! per unit weight is one standard deviation of the ANSUR II variable, measured on the template
//! ([`measure`], `data/anthropometry/ansur2.json`; a US Army population, see there). Every other
//! amount is a model choice made by reading the rendered result, stated per unit weight in the
//! recipe's documentation, and is not a statistic. The registry keeps its provisional ranges (span
//! ±3, usual ±2) until a statistical model recalibrates them.
//!
//! **By name only.** Recipes look up landmarks, joints, regions and mesh parts by their stable
//! names, never by vertex index or coordinate, so they follow the generator's body changes. A
//! template missing a name fails with [`IdentityError::Missing`].
//!
//! - [`error`]: the error type, [`IdentityError`].
//! - [`space`]: the surface context (part kinds, mirror map, confinement, smoothing, finishing).
//! - [`skeletal`]: bone stretches that move joints and skin together.
//! - [`measure`]: measurements on the template and the ANSUR II calibration.
//! - [`body`]: global axes, body proportions, torso and breast shape, hands and feet.
//! - [`face`]: face, ears, neck, eyes, teeth and the asymmetry components.
//! - [`scan`]: the face-shape components measured from scans (scan-derived, not authored).
//! - [`sex`]: the Female / Male switch `identity.sex` (body, face and genital variant).
//!
//! Entry points: [`build`] returns the components with their regressor rows; [`with_identity`]
//! returns a new template that has them (existing components are kept and re-indexed by name).

pub mod body;
pub mod error;
pub mod face;
pub mod measure;
pub mod scan;
pub mod sex;
pub mod skeletal;
pub mod space;

use humanart_math::Vec3;

pub use error::{IdResult, IdentityError};
use skeletal::Bones;
pub use space::{IdentityComponent, Reach, Space, Symmetry};

use crate::delta::{DeltaBases, DeltaBasesBuilder};
use crate::skeleton::JointRegressor;
use crate::template::{ProvenanceEntry, Template};

/// Registry parameter id and the component it binds to (`[REG-01]` `evaluation: component`), in
/// registry order: `data/template/params/{identity,body,face,eyes,mouth}.json`.
pub const BINDINGS: &[(&str, &str)] = &[
    ("identity.age", "global.age"),
    ("identity.mass", "global.mass"),
    ("identity.muscularity", "global.muscularity"),
    ("identity.height", "global.height"),
    ("identity.sex_shape", "global.sex"),
    ("identity.asymmetry", "asymmetry.global"),
    ("identity.sex", sex::SWITCH),
    ("body.proportions.arm_length", "semantic.arm-length"),
    ("body.proportions.leg_length", "semantic.leg-length"),
    ("body.proportions.shoulder_width", "semantic.shoulder-width"),
    ("body.proportions.hip_width", "semantic.hip-width"),
    ("body.hands.palm_width", "semantic.palm-width"),
    ("body.hands.finger_length", "semantic.finger-length"),
    ("body.hands.finger_length_ratio", "semantic.finger-ratio"),
    ("body.hands.knuckle_prominence", "semantic.knuckle-prominence"),
    ("body.feet.length", "semantic.foot-length"),
    ("body.feet.width", "semantic.foot-width"),
    ("body.torso.bust_size", "semantic.bust-size"),
    ("body.torso.breast_projection", "semantic.breast-projection"),
    ("body.torso.breast_ptosis", "semantic.breast-ptosis"),
    ("body.torso.breast_spacing", "semantic.breast-spacing"),
    ("body.torso.nipple_size", "semantic.nipple-size"),
    ("body.torso.areola_size", "semantic.areola-size"),
    ("body.torso.waist_width", "semantic.waist-width"),
    ("body.torso.glute_size", "semantic.glute-size"),
    ("body.legs.hip_flank", "semantic.hip-flank"),
    ("body.legs.thigh_fullness", "semantic.thigh-fullness"),
    ("face.nose.bridge_height", "semantic.nose-bridge-height"),
    ("face.nose.bridge_width", "semantic.nose-bridge-width"),
    ("face.nose.length", "semantic.nose-length"),
    ("face.nose.tip_projection", "semantic.nose-tip-projection"),
    ("face.nose.tip_rotation", "semantic.nose-tip-rotation"),
    ("face.nose.nostril_width", "semantic.nostril-width"),
    ("face.eyes.spacing", "semantic.eye-spacing"),
    ("face.eyes.size", "semantic.eye-size"),
    ("face.eyes.canthal_tilt", "semantic.canthal-tilt"),
    ("face.eyes.depth", "semantic.eye-depth"),
    ("face.eyelids.crease_height", "semantic.lid-crease-height"),
    ("face.eyelids.hooding", "semantic.lid-hooding"),
    ("face.brow.ridge", "semantic.brow-ridge"),
    ("face.brow.height", "semantic.brow-height"),
    ("face.forehead.height", "semantic.forehead-height"),
    ("face.forehead.slope", "semantic.forehead-slope"),
    ("face.lips.upper_fullness", "semantic.upper-lip-fullness"),
    ("face.lips.lower_fullness", "semantic.lower-lip-fullness"),
    ("face.lips.cupids_bow", "semantic.cupids-bow"),
    ("face.mouth.width", "semantic.mouth-width"),
    ("face.mouth.protrusion", "semantic.mouth-protrusion"),
    ("face.philtrum.length", "semantic.philtrum-length"),
    ("face.jaw.width", "semantic.jaw-width"),
    ("face.jaw.angle", "semantic.jaw-angle"),
    ("face.chin.projection", "semantic.chin-projection"),
    ("face.chin.height", "semantic.chin-height"),
    ("face.chin.width", "semantic.chin-width"),
    ("face.cheeks.cheekbone_prominence", "semantic.cheekbone-prominence"),
    ("face.cheeks.fullness", "semantic.cheek-fullness"),
    ("face.ears.size", "semantic.ear-size"),
    ("face.ears.protrusion", "semantic.ear-protrusion"),
    ("face.ears.lobe_size", "semantic.earlobe-size"),
    ("face.neck.width", "semantic.neck-width"),
    ("face.neck.length", "semantic.neck-length"),
    ("face.nose.asymmetry", "asymmetry.nose"),
    ("face.eyes.asymmetry", "asymmetry.eyes"),
    ("face.mouth.asymmetry", "asymmetry.mouth"),
    ("face.jaw.asymmetry", "asymmetry.jaw"),
    ("face.ears.asymmetry", "asymmetry.ears"),
    ("face.brow.asymmetry", "asymmetry.brow"),
    ("face.cheeks.asymmetry", "asymmetry.cheeks"),
    ("eyes.eyeball.size", "semantic.eyeball-size"),
    ("mouth.teeth.incisor_proportion", "semantic.teeth-incisor"),
    ("mouth.teeth.canine_proportion", "semantic.teeth-canine"),
    ("mouth.teeth.molar_proportion", "semantic.teeth-molar"),
    ("mouth.gums.line_height", "semantic.gum-line"),
];

/// What the recipes share: the surface context and the bones.
#[derive(Clone, Debug)]
pub struct Kit<'s, 't> {
    /// The surface context.
    pub space: &'s Space<'t>,
    /// The skeleton's bones.
    pub bones: Bones,
}

impl<'s, 't> Kit<'s, 't> {
    /// The kit of `space`.
    pub fn new(space: &'s Space<'t>) -> IdResult<Self> {
        Ok(Self { space, bones: Bones::new(space)? })
    }
}

/// A confined displacement field (per node) with joint offsets and what it may move: the
/// building block that components are summed from.
#[derive(Clone, Debug)]
pub struct Raw {
    /// Per-node displacement per unit weight, metres.
    pub disp: Vec<Vec3>,
    /// Joint offsets per unit weight, metres (any order, repeated joints add up).
    pub joints: Vec<(u16, Vec3)>,
    /// What it may move.
    pub reaches: Vec<Reach<'static>>,
}

impl Raw {
    /// No displacement.
    #[must_use]
    pub fn zero(space: &Space<'_>) -> Self {
        Self { disp: vec![Vec3::ZERO; space.node_count()], joints: Vec::new(), reaches: Vec::new() }
    }

    /// Confines `disp` to `reach` and wraps it.
    pub fn confined(
        space: &Space<'_>,
        reach: Reach<'static>,
        symmetry: Symmetry,
        mut disp: Vec<Vec3>,
        joints: Vec<(u16, Vec3)>,
    ) -> IdResult<Self> {
        space.confine(&reach, symmetry, &mut disp)?;
        Ok(Self { disp, joints, reaches: vec![reach] })
    }

    /// Adds `s` times `other`.
    pub fn add(&mut self, other: &Self, s: f32) {
        for (d, o) in self.disp.iter_mut().zip(&other.disp) {
            *d += *o * s;
        }
        self.joints.extend(other.joints.iter().map(|&(j, d)| (j, d * s)));
        self.reaches.extend(other.reaches.iter().copied());
    }

    /// This field times `s`.
    #[must_use]
    pub fn scaled(mut self, s: f32) -> Self {
        for d in &mut self.disp {
            *d *= s;
        }
        for j in &mut self.joints {
            j.1 *= s;
        }
        self
    }

    /// The finished component called `name`.
    /// Every component except nipple size first carries the nipples and the navel rigidly
    /// ([`Space::carry_details`]).
    pub fn component(mut self, space: &Space<'_>, name: &str, symmetry: Symmetry) -> IdResult<IdentityComponent> {
        if name != "semantic.nipple-size" {
            space.carry_details(&mut self.disp);
        }
        space.component(name, symmetry, &self.reaches, self.disp, self.joints)
    }
}

/// Builds the component bound to `component` (a [`BINDINGS`] target).
pub fn recipe(kit: &Kit<'_, '_>, component: &str) -> IdResult<IdentityComponent> {
    if component == sex::SWITCH {
        return sex::switch(kit);
    }
    if let Some(c) = body::recipe(kit, component)? {
        return Ok(c);
    }
    if let Some(c) = face::recipe(kit, component)? {
        return Ok(c);
    }
    Err(IdentityError::missing("identity recipe", component))
}

/// Builds every component of [`BINDINGS`] for `template`, in binding order.
pub fn build(template: &Template) -> IdResult<Vec<IdentityComponent>> {
    let space = Space::new(template)?;
    let kit = Kit::new(&space)?;
    BINDINGS.iter().map(|&(_, c)| recipe(&kit, c)).collect()
}

/// Per component: name, stored entries, largest displacement per unit weight (metres), joints
/// moved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdentityReport {
    /// One line per component, in binding order.
    pub components: Vec<(String, usize, f32, usize)>,
}

/// A new template with the identity components of [`build`] added to `template`'s delta bases and
/// joint regressor (see [`add_components`]), and a report of what was added.
pub fn with_identity(template: &Template) -> IdResult<(Template, IdentityReport)> {
    let components = build(template)?;
    let mut report = IdentityReport::default();
    for comp in &components {
        let max = comp.entries.iter().map(|e| e.1.length()).fold(0.0, f32::max);
        report.components.push((comp.name.clone(), comp.entries.len(), max, comp.joints.len()));
    }
    Ok((add_components(template, &components)?, report))
}

/// A new template with `components` added to `template`'s delta bases and joint regressor.
/// Existing components (expressions, correctives) are kept and every index that refers to them
/// (expression set, regressor rows) is re-mapped by name. Fails when the template already has a
/// component of the same name.
pub fn add_components(template: &Template, components: &[IdentityComponent]) -> IdResult<Template> {
    let old = template.deltas();
    if let Some(c) = components.iter().position(|c| old.find(&c.name).is_some()) {
        return Err(IdentityError::invalid(
            "identity component",
            c,
            "the template already has a component of this name",
        ));
    }
    rebuild(template, |_| true, components)
}

/// A new template without the components [`BINDINGS`] names (the identity components of
/// [`with_identity`]); everything else is kept and re-indexed by name, as in [`add_components`].
/// The template's provenance entry for them stays. Used to rebuild or re-calibrate the identity
/// components of a template that already has them.
pub fn without_identity(template: &Template) -> IdResult<Template> {
    rebuild(template, |name| !BINDINGS.iter().any(|b| b.1 == name), &[])
}

/// The template with only the existing components `keep` accepts, every index that refers to a
/// component re-mapped by name (definitions that point at a dropped component must be removed
/// first).
pub(crate) fn rebuild_keeping(template: &Template, keep: impl Fn(&str) -> bool) -> IdResult<Template> {
    rebuild(template, keep, &[])
}

/// The template with the existing components `keep` accepts plus `components`, every index that
/// refers to a component re-mapped by name.
fn rebuild(template: &Template, keep: impl Fn(&str) -> bool, components: &[IdentityComponent]) -> IdResult<Template> {
    let old = template.deltas();
    let mut builder = DeltaBasesBuilder::new();
    for (c, comp) in old.components().iter().enumerate() {
        if !keep(&comp.name) {
            continue;
        }
        let entries: Vec<(u32, Vec3)> =
            old.indices(c).iter().zip(old.deltas(c)).map(|(&v, &d)| (v, DeltaBases::delta_f32(d))).collect();
        builder.add(&comp.name, comp.kind, comp.region, &entries)?;
    }
    for comp in components {
        builder.add(&comp.name, comp.kind, comp.region, &comp.entries)?;
    }
    let deltas = builder.finish()?;
    let find = |name: &str| deltas.find(name).ok_or_else(|| IdentityError::missing("delta component", name));
    let mut parts = template.clone().into_parts();
    // Re-index everything that points at a component.
    let old_name = |c: u32| old.components().get(c as usize).map_or("", |x| x.name.as_str());
    for e in &mut parts.expressions.expressions {
        e.component = find(old_name(e.component))?;
    }
    for e in &mut parts.expressions.combinations {
        e.component = find(old_name(e.component))?;
    }
    for e in &mut parts.expressions.pose_correctives {
        e.component = find(old_name(e.component))?;
    }
    let mut rows: Vec<Vec<(u16, Vec3)>> = vec![Vec::new(); deltas.len()];
    for (c, comp) in old.components().iter().enumerate() {
        if !keep(&comp.name) {
            continue;
        }
        let (j, d) = template.skeleton().regressor.component(c);
        if let Some(row) = rows.get_mut(find(&comp.name)? as usize) {
            *row = j.iter().copied().zip(d.iter().copied()).collect();
        }
    }
    for comp in components {
        if let Some(row) = rows.get_mut(find(&comp.name)? as usize) {
            row.clone_from(&comp.joints);
        }
    }
    let mut regressor = JointRegressor { offsets: vec![0], joints: Vec::new(), deltas: Vec::new() };
    for row in rows {
        for (j, d) in row {
            regressor.joints.push(j);
            regressor.deltas.push(d);
        }
        regressor.offsets.push(regressor.joints.len() as u32);
    }
    parts.skeleton.regressor = regressor;
    parts.deltas = deltas;
    let entry = ProvenanceEntry {
        item: "identity axes, semantic and asymmetry components with their joint-regressor rows (WP-06, ID-07)".into(),
        source: "authored procedurally by humanart-model identity::{body, face}; amounts are model choices, not measurements".into(),
        licence: parts.info.licence.clone(),
    };
    if !parts.info.provenance.contains(&entry) {
        parts.info.provenance.push(entry);
    }
    Ok(Template::new(parts)?)
}

#[cfg(test)]
mod tests;
