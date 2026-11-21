//! The compositor's baked input layer (todo 1.4): the template bakers' thickness (`[BAKE-01]`,
//! metres, `f32`) and ambient occlusion (`[BAKE-02]`, 0 closed .. 1 open, `f32`) converted to the
//! RGBA8 texels `SkinCompositor::upload_inputs` takes as its first layer:
//!
//! - red: thickness in units of [`THICKNESS_UNIT_MM`] (0.1 mm), so 0..25.4 mm; [`NO_THICKNESS`]
//!   (255) means "no baked value" and the shading falls back to the shadow-map thickness
//!   (`shaders/render/skin_surface.wgsl`, `BAKED_THICKNESS_RANGE_MM` = 255 units);
//! - green: the occlusion, `round(255 · ao)` (255 = open), multiplied into the resolved cavity
//!   that occludes the environment light and the specular (`[SHD-08]`);
//! - blue: the low-frequency tone variation (`super::variation`), `round(255 · v)`, 128 neutral;
//! - alpha: 0 (unused).

use super::SkinError;

/// Millimetres per unit of the thickness byte.
pub const THICKNESS_UNIT_MM: f32 = 0.1;
/// The thickness byte of a texel without a baked value.
pub const NO_THICKNESS: u8 = 255;

/// The thickness byte of `metres` (non-finite or negative: [`NO_THICKNESS`]; above 25.4 mm the
/// largest value, 254).
pub fn thickness_byte(metres: f32) -> u8 {
    if !metres.is_finite() || metres < 0.0 {
        return NO_THICKNESS;
    }
    let units = (metres * 1000.0 / THICKNESS_UNIT_MM).round();
    // In 0..=254 after the clamp, so the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = units.clamp(0.0, f32::from(NO_THICKNESS - 1)) as u8;
    byte
}

/// The occlusion byte of `open` (1 open, 0 closed; non-finite: open).
pub fn occlusion_byte(open: f32) -> u8 {
    let v = if open.is_finite() { open.clamp(0.0, 1.0) } else { 1.0 };
    // In 0..=255 after the clamp.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = (v * 255.0).round() as u8;
    byte
}

/// Writes the baked input layer of one tile into `out` (cleared first): `texels` RGBA8 texels from
/// one thickness (metres; NaN where nothing was baked), one occlusion value and one tone variation
/// value per texel, any map missing (`None`: no thickness, open, neutral).
///
/// # Errors
/// [`SkinError::Invalid`] when a given map does not hold `texels` values.
pub fn baked_inputs(
    texels: usize,
    thickness_m: Option<&[f32]>,
    occlusion: Option<&[f32]>,
    variation: Option<&[f32]>,
    out: &mut Vec<u8>,
) -> Result<(), SkinError> {
    if thickness_m.is_some_and(|t| t.len() != texels)
        || occlusion.is_some_and(|o| o.len() != texels)
        || variation.is_some_and(|v| v.len() != texels)
    {
        return Err(SkinError::Invalid { what: "a baked map does not cover the tile" });
    }
    out.clear();
    out.reserve(texels * 4);
    for i in 0..texels {
        let t = thickness_m.and_then(|t| t.get(i)).map_or(NO_THICKNESS, |&m| thickness_byte(m));
        let o = occlusion.and_then(|o| o.get(i)).map_or(255, |&a| occlusion_byte(a));
        let v = variation.and_then(|v| v.get(i)).map_or(128, |&a| occlusion_byte(if a.is_finite() { a } else { 0.5 }));
        out.extend_from_slice(&[t, o, v, 0]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_1_4_baked_thickness_metres_to_the_compositor_byte() {
        // [BAKE-01] metres -> 0.1 mm units; the shader reads byte / 255 * 25.5 mm.
        assert_eq!(thickness_byte(0.0), 0);
        assert_eq!(thickness_byte(0.0012), 12, "1.2 mm");
        assert_eq!(thickness_byte(0.00125), 13, "rounded");
        assert_eq!(thickness_byte(0.1), 254, "clamped below the none value");
        assert_eq!(thickness_byte(f32::NAN), NO_THICKNESS);
        assert_eq!(thickness_byte(-1.0), NO_THICKNESS);
        let shader_mm = |b: u8| f32::from(b) / 255.0 * 25.5;
        assert!((shader_mm(thickness_byte(0.0034)) - 3.4).abs() < 1e-4);
        assert_eq!(occlusion_byte(1.0), 255);
        assert_eq!(occlusion_byte(0.5), 128);
        assert_eq!(occlusion_byte(-3.0), 0);
        assert_eq!(occlusion_byte(f32::INFINITY), 255);
        let mut out = Vec::new();
        baked_inputs(2, Some(&[0.002, f32::NAN]), Some(&[0.0, 1.0]), Some(&[0.0, 1.0]), &mut out).unwrap();
        assert_eq!(out, [20, 0, 0, 0, 255, 255, 255, 0]);
        baked_inputs(1, None, None, None, &mut out).unwrap();
        assert_eq!(out, [255, 255, 128, 0]);
        assert!(baked_inputs(3, Some(&[0.0]), None, None, &mut out).is_err());
        assert!(baked_inputs(1, None, None, Some(&[0.5, 0.5]), &mut out).is_err());
    }
}
