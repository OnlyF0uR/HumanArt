// mouth/mouth.wgsl: teeth, gums and tongue in the geometry pass [MTH-01]..[MTH-04]. See
// crates/render/src/mouth/mod.rs for the entry layout (`MouthShading::entry`).
//
// Needs eye/features.wgsl and what it needs (lib/math, lib/brdf, render/frame, render/lighting)
// included before it.
//
// Every point inside the mouth is darkened by the lip opening's form factor (mouth_occlusion, the
// twin of humanart_model::mouth::MouthOpening::occlusion): the ambient light by the factor, the
// direct light by a softer function of it (the shadow maps miss the lips' millimetre-scale
// shadowing), so the interior is dark behind a narrow opening and never looks lit and hollow
// [MTH-04]. The mouth-bag skin (MOUTH_BAG_CLASS: the body's mouth-interior faces, drawn as their
// own object) is shaded here too, as wet oral mucosa under the same cavity occlusion, so the
// throat and cheeks behind the teeth go dark the way a real mouth does.

// The fraction of light reaching `p` through the lip opening of entry `slot` (1 in front of it):
// the form factor A / (A + pi d^2) of the opening's area A seen from depth d, the lateral
// distance beyond the ellipse's rim added to the depth.
fn mouth_occlusion(slot: u32, p: vec3<f32>) -> f32 {
    let c = feature_vec(slot, 0u);
    let f = feature_vec(slot, 1u);
    let a = feature_vec(slot, 2u);
    let q = p - c.xyz;
    let depth = -dot(q, f.xyz);
    if (depth <= 0.0) {
        return 1.0;
    }
    let up = cross(f.xyz, a.xyz);
    let x = dot(q, a.xyz);
    let y = dot(q, up);
    let e = sqrt(sq(x / max(c.w, 1e-5)) + sq(y / max(f.w, 1e-5)));
    let outside = select(0.0, sqrt(x * x + y * y) * (1.0 - 1.0 / e), e > 1.0);
    let area = PI * c.w * f.w;
    let d2 = depth * depth + outside * outside;
    return area / max(area + PI * d2, 1e-12);
}

// Teeth [MTH-01]: gingival-to-incisal gradient, translucent incisal edge, staining, wet enamel.
fn mouth_teeth(slot: u32, world: vec3<f32>, ng: vec3<f32>, v: vec3<f32>, light: vec2<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let axes = feature_vec(slot, 3u);
    let crown = max(feature_vec(slot, 2u).w, 1e-4);
    let gingival = feature_vec(slot, 4u);
    let incisal = feature_vec(slot, 5u);
    let finish = feature_vec(slot, 6u);
    // 0 at the incisal edge of the arch the point belongs to, 1 at the neck.
    let y = dot(world, axes.xyz);
    let lower_edge = finish.w;
    let h = saturate(select(lower_edge - y, y - axes.w, y > 0.5 * (axes.w + lower_edge)) / crown);
    var albedo = mix(incisal.rgb, gingival.rgb, smoothstep(0.0, 0.85, h));
    // The translucent edge shows the dark mouth behind it: greyer, bluish, less when worn.
    let edge = (1.0 - smoothstep(0.0, 0.16, h)) * gingival.w * (1.0 - 0.6 * incisal.w);
    albedo = mix(albedo, albedo * vec3<f32>(0.55, 0.6, 0.7), edge);
    // Stain near the gum line and in patches.
    let patches = feature_noise(world * 900.0);
    albedo *= 0.94 + 0.12 * feature_noise(world * 140.0 + 3.7);
    albedo *= mix(1.0, 0.78, sq(1.0 - saturate(dot(ng, v))));
    let stain = finish.x * (0.8 * smoothstep(0.55, 1.0, h) + 0.35 * patches);
    albedo = mix(albedo, albedo * vec3<f32>(0.86, 0.72, 0.5), saturate(stain));
    let rough = mix(0.42, 0.14, finish.y);
    // Enamel is translucent: light wraps further round the thin incisal edge than round the neck.
    let wrap = gingival.w * (0.3 + 0.5 * (1.0 - smoothstep(0.0, 0.35, h)));
    var out = feature_lit(world, ng, ng, v, rough, finish.z, wrap, light.y, light.x, pixel, taps);
    out.albedo = albedo;
    // Light through the edge from behind (the lit cavity side) glows faintly warm.
    out.diffuse += out.diffuse * (edge * 0.25);
    return out;
}

// Gums [MTH-02]: scattering (wrapped) diffuse with a slight stippling, wet specular.
fn mouth_gums(slot: u32, world: vec3<f32>, ng: vec3<f32>, v: vec3<f32>, light: vec2<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let gum = feature_vec(slot, 7u);
    let wet = feature_vec(slot, 8u).x;
    let stipple = feature_noise(world * 1500.0);
    let rough = mix(0.55, 0.2, wet);
    var out = feature_lit(world, ng, ng, v, rough, 0.02, 0.5 * gum.w, light.y, light.x, pixel, taps);
    out.albedo = gum.rgb * (0.9 + 0.2 * stipple);
    return out;
}

// Tongue [MTH-03]: coating on the dorsum, papillae micro-normal, broad wet specular.
fn mouth_tongue(slot: u32, world: vec3<f32>, ng: vec3<f32>, v: vec3<f32>, light: vec2<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let up = feature_vec(slot, 3u).xyz;
    let surface = feature_vec(slot, 8u);
    let color = feature_vec(slot, 9u);
    // Papillae: the gradient of 0.4 mm noise tilts the normal.
    let scale = 2500.0;
    let e = 0.25 / scale;
    let n0 = feature_noise(world * scale);
    let g = vec3<f32>(
        feature_noise((world + vec3<f32>(e, 0.0, 0.0)) * scale) - n0,
        feature_noise((world + vec3<f32>(0.0, e, 0.0)) * scale) - n0,
        feature_noise((world + vec3<f32>(0.0, 0.0, e)) * scale) - n0,
    ) * 4.0;
    let tangential = g - ng * dot(g, ng);
    let n = safe_normalize(ng - tangential * (0.16 * surface.w), ng);
    let dorsum = saturate(dot(ng, up) * 2.0);
    let coat = saturate(surface.z * dorsum * (0.3 + 0.7 * feature_fbm(world * 250.0)));
    let albedo = mix(color.rgb, vec3<f32>(0.72, 0.62, 0.60), coat) * (0.94 + 0.12 * n0);
    let rough = mix(0.75, 0.42, color.w);
    var out = feature_lit(world, n, ng, v, rough, 0.02, 0.45, light.y, light.x, pixel, taps);
    out.albedo = albedo;
    // Between the papillae the film is shadowed: the sheen breaks into small glints.
    out.specular *= 0.3 + 0.4 * smoothstep(0.3, 0.8, n0);
    return out;
}

// The mouth-bag skin: wet oral mucosa (the gum colour, deeper), soft scattering, its own darker
// cavity falloff on top of the opening's form factor (the throat is the deepest point).
fn mouth_bag(slot: u32, world: vec3<f32>, ng: vec3<f32>, v: vec3<f32>, light: vec2<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let gum = feature_vec(slot, 7u);
    let wet = feature_vec(slot, 8u).x;
    let rough = mix(0.5, 0.22, wet);
    let deeper = vec2<f32>(light.x * mix(0.5, 1.0, light.x), light.y * sqrt(light.x));
    var out = feature_lit(world, ng, ng, v, rough, 0.02, 0.5, deeper.y, deeper.x, pixel, taps);
    out.albedo = gum.rgb * vec3<f32>(0.85, 0.62, 0.64) * (0.9 + 0.2 * feature_noise(world * 700.0));
    return out;
}

// The mouth's shading at a pixel of the teeth, gums, tongue or mouth bag (by material class) of entry `slot`:
// mesh point `world` with geometric normal `ng`, seen from `camera`.
fn mouth_shade(slot: u32, part_class: u32, world: vec3<f32>, ng: vec3<f32>, camera: vec3<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let v = safe_normalize(camera - world, ng);
    let o = mouth_occlusion(slot, world);
    // Ambient (x) and direct (y) light factors [MTH-04].
    let light = vec2<f32>(o, smoothstep(0.0, 0.6, o));
    if (part_class == TEETH_CLASS) {
        return mouth_teeth(slot, world, ng, v, light, pixel, taps);
    }
    if (part_class == TONGUE_CLASS) {
        return mouth_tongue(slot, world, ng, v, light, pixel, taps);
    }
    if (part_class == MOUTH_BAG_CLASS) {
        return mouth_bag(slot, world, ng, v, light, pixel, taps);
    }
    return mouth_gums(slot, world, ng, v, light, pixel, taps);
}
