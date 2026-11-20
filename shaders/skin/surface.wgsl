// skin/surface.wgsl: shading helpers for the skin passes [SKN-21] [SKN-31] [SKN-32], twins of
// humanart_model::skin_appearance::{microdetail, zones}.

// Cavity-driven micro-shadowing [SHD-09]: saturate(|N.L| + 2 ao^2 - 1) (D. Chan, "Material
// Advances in Call of Duty: WWII", SIGGRAPH 2018 Advances course).
fn skin_micro_shadow(ao: f32, n_dot_l: f32) -> f32 {
    return clamp(abs(n_dot_l) + 2.0 * ao * ao - 1.0, 0.0, 1.0);
}

// Horizon-based self-shadowing: visibility of a light at elevation sine `light_sin` against the
// horizon sine at its azimuth.
fn skin_horizon_visibility(horizon_sin: f32, light_sin: f32, softness: f32) -> f32 {
    return clamp((light_sin - horizon_sin) / max(softness, 1e-4) + 0.5, 0.0, 1.0);
}

// The flushed albedo [SKN-31]: the resolved albedo plus the stored full-flush delta times the
// animated flush amount.
fn skin_flushed_albedo(albedo: vec3<f32>, flush_delta: vec3<f32>, flush: f32) -> vec3<f32> {
    return max(albedo + flush_delta * clamp(flush, 0.0, 1.0), vec3<f32>(0.0));
}
