// eye/eye.wgsl: the eye in the geometry pass [EYE-01]..[EYE-05]. The eyeball part's pixels are
// shaded by tracing the camera ray against the analytic eye of their table entry (eye/optics.wgsl):
// the sclera, or the cornea and, refracted through it, the iris plane [EYE-02] [AC-WP16-2]. See
// crates/render/src/eye/mod.rs for the entry layout (`EyeShading::entry`).
//
// Needs eye/features.wgsl, eye/optics.wgsl and what they need (lib/math, lib/brdf, lib/sampling,
// render/frame, render/lighting) included before it.
//
// One object draws both eyeballs (and one both tear lines, one both occlusion shells): a pixel
// shades with its object's entry or the entry's partner, whichever eyeball centre is nearer
// (eye_pick); the partner's side is the medial side.
//
// - Sclera [EYE-03]: base colour yellowed with age, a procedural network of conjunctival vessels
//   (domain-warped ridged noise on the sphere) denser toward the canthi and fading at the limbus,
//   limbal darkening, the caruncle and the plica semilunaris at the medial canthus, wrapped
//   (subsurface-soft) diffuse, and a wet tear film: a near-mirror specular whose normal follows
//   the conjunctiva's fine relief, so highlights break up the way they do on a real eye.
// - Cornea and iris [EYE-02] [EYE-04]: a clear specular layer (tear film f0, near-mirror), so the
//   environment and the lights show as catchlights; behind it the iris plane seen through both
//   corneal surfaces. The iris radial coordinate runs from the pupil margin to the limbus, so the
//   layers stretch with the pupil: radial fibres, collarette ridge, crypts, pupillary ruff,
//   contraction furrows, limbal ring. Light reaches the iris refracted by the cornea and reduced
//   by its Fresnel transmission; the opaque limbus shades the iris rim on the light's side (the
//   refracted light ray is followed back up to the limbus plane), which, with the parallax, is
//   what makes the iris read as lying deep under a clear dome.
// - Occlusion [EYE-05]: the lid aperture (angles about the eyeball centre in the head frame)
//   gives the eyeball a contact occlusion right at the margins and the shadow of the upper lid
//   and its lashes on the direct light (the lash density is the entry's lash hook). The broad,
//   soft eye-occlusion shadow is a translucent film (eye_film: its transmittance from the lid
//   aperture) drawn by the occlusion shell (EYE_OCCLUSION_CLASS) over the eyeball. The geometry
//   pass has no blending, so a shell pixel composites itself: it traces the eyeball behind it
//   and returns that shading times the film's transmittance; an eyeball pixel the shell does not
//   cover gets the same film, so the shadow is continuous across the shell's rim.
// - Tear line (TEAR_CLASS) [EYE-04]: the lacrimal meniscus, a clear liquid strip composited the
//   same way: the eyeball behind it, transmitted through the liquid's Fresnel, plus the strip's
//   own sharp reflection; where the ray past it misses the eyeball (over the lid margin) it is a
//   wet strip over the margin's colour.

// Reflectance at normal incidence of the tear film (n = 1.337, humanart_model::eye::params).
const EYE_TEAR_F0: f32 = 0.0208;
// Perceptual roughness of the cornea's tear film (near mirror; catchlights stay a few pixels).
const EYE_CORNEA_ROUGHNESS: f32 = 0.05;
// Albedo of the pupil (the dark fundus seen through it).
const EYE_PUPIL_ALBEDO: vec3<f32> = vec3<f32>(0.010, 0.008, 0.008);
// Largest opacity of the eye-occlusion film (at the lid margins).
const EYE_OCCLUSION_OPACITY: f32 = 0.7;
// Caruncle and plica colours (moist pink mucosa, linear).
const EYE_CARUNCLE: vec3<f32> = vec3<f32>(0.62, 0.27, 0.25);
const EYE_PLICA: vec3<f32> = vec3<f32>(0.70, 0.45, 0.42);

// The entry of the eye nearer to `p`: `slot` or its partner (float 39: partner entry + 1).
fn eye_pick(slot: u32, p: vec3<f32>) -> u32 {
    let partner = u32(max(feature_float(slot, 39u), 0.0) + 0.5);
    if (partner == 0u || partner > MAX_FEATURE_ENTRIES || feature_kind(partner) != FEATURE_EYE) {
        return slot;
    }
    let own = feature_vec(slot, 0u).xyz;
    let other = feature_vec(partner, 0u).xyz;
    return select(slot, partner, distance(p, other) < distance(p, own));
}

// The analytic eye of entry `slot`.
fn eye_shape_of(slot: u32) -> EyeShape {
    let c = feature_vec(slot, 4u);
    return eye_shape(feature_vec(slot, 0u), feature_vec(slot, 1u), feature_vec(slot, 2u), feature_vec(slot, 3u), c.xy);
}

// Point `p` in the head frame of entry `slot`, relative to the eyeball centre.
fn eye_head_local(slot: u32, centre: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let q = feature_vec(slot, 10u);
    return quat_rotate(vec4<f32>(-q.xyz, q.w), p - centre);
}

// The lid aperture at point `p` of the eye: x = signed angular distance inside the opening
// (positive inside: the distance to the nearer lid or canthus), y = angular distance below the
// upper lid, z = angular distance from the medial canthus (large when the side is unknown).
// Radians; mirrors humanart_model::eye::LidAperture::openness.
fn eye_lid_openness(slot: u32, centre: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let d = eye_head_local(slot, centre, p);
    let azimuth = atan2(d.x, d.z);
    let elevation = atan2(d.y, length(d.xz));
    let first = feature_float(slot, 44u);
    let last = feature_float(slot, 45u);
    let x = clamp((azimuth - first) / max(last - first, 1e-6), 0.0, 1.0) * 7.0;
    let k = min(u32(floor(x)), 6u);
    let f = x - f32(k);
    let upper = mix(feature_float(slot, 46u + k), feature_float(slot, 47u + k), f);
    let lower = mix(feature_float(slot, 54u + k), feature_float(slot, 55u + k), f);
    let vertical = min(upper - elevation, elevation - lower);
    let horizontal = min(azimuth - first, last - azimuth);
    // The medial side is toward the partner eye (in the head frame).
    var medial = 10.0;
    let partner = u32(max(feature_float(slot, 39u), 0.0) + 0.5);
    if (partner != 0u && partner <= MAX_FEATURE_ENTRIES) {
        let toward = eye_head_local(slot, centre, feature_vec(partner, 0u).xyz).x;
        medial = select(last - azimuth, azimuth - first, toward < 0.0);
    }
    return vec3<f32>(min(vertical, horizontal), upper - elevation, medial);
}

// Ambient (x) and direct (y) light factors on the eyeball for openness `o` and lash density
// `lashes`: a contact occlusion at the margins (the soft shadow is the occlusion film's), the
// shadow of the upper lid's thickness and of its lashes on the direct light.
fn eye_lid_factors(o: vec3<f32>, lashes: f32) -> vec2<f32> {
    let inside = smoothstep(-0.03, 0.03, o.x);
    let ambient = mix(0.45, 1.0, smoothstep(-0.01, 0.07, o.x)) * inside;
    let lid = mix(0.45, 1.0, smoothstep(0.0, 0.08, o.y));
    let lash = 1.0 - saturate(lashes) * 0.55 * (1.0 - smoothstep(0.03, 0.2, o.y));
    return vec2<f32>(ambient, inside * lid * lash);
}

// The sclera's albedo at point `p` on the eyeball, `medial` radians from the medial canthus
// [EYE-03].
fn eye_sclera_albedo(slot: u32, s: EyeShape, p: vec3<f32>, medial: f32) -> vec3<f32> {
    let base = feature_vec(slot, 8u);
    let looks = feature_vec(slot, 9u);
    let seed = feature_float(slot, 31u);
    let local = (p - s.centre) / max(s.radius, 1e-6);
    let h = dot(local, s.axis);
    let limbus_h = s.limbus_z / max(s.radius, 1e-6);
    // 1 at the limbus, 0 a quarter radius away from it.
    let near_limbus = 1.0 - smoothstep(0.0, 0.25, limbus_h - h);
    let side = abs(dot(local, cross(s.up, s.axis)));
    var albedo = base.rgb;
    // Age yellowing, strongest away from the cornea (pinguecula zones toward the canthi).
    albedo = mix(albedo, albedo * vec3<f32>(1.0, 0.9, 0.66), looks.y * (1.0 - 0.6 * near_limbus));
    // Faint blue-grey of the choroid showing through the thin sclera near the limbus.
    albedo = mix(albedo, albedo * vec3<f32>(0.9, 0.93, 1.0), 0.5 * near_limbus);
    // Conjunctival vessels: ridges of domain-warped noise, thin and thinner branches.
    // The vessels run from the fornices toward the limbus: the noise is stretched along the
    // meridian (the direction toward the cornea) so its ridges do too.
    let meridian = safe_normalize(s.axis - local * h, s.up);
    let stretched = local - meridian * (0.6 * dot(local, meridian));
    let q = stretched * 7.0 + vec3<f32>(seed * 0.37, seed * 0.11, seed * 0.23);
    let warp = vec3<f32>(feature_noise(q * 1.7 + 3.1), feature_noise(q * 1.7 + 11.3), feature_noise(q * 1.7 + 23.7)) - vec3<f32>(0.5);
    let n1 = feature_noise(q + warp * 1.4);
    let n2 = feature_noise(q * 2.7 + warp * 2.2 + 7.0);
    let n3 = feature_noise(q * 6.1 + warp * 3.0 + 19.0);
    let vessels = pow(1.0 - abs(2.0 * n1 - 1.0), 28.0) + 0.6 * pow(1.0 - abs(2.0 * n2 - 1.0), 40.0) + 0.35 * pow(1.0 - abs(2.0 * n3 - 1.0), 60.0);
    let density = base.w * mix(0.3, 1.0, smoothstep(0.2, 0.75, side)) * smoothstep(0.03, 0.4, limbus_h - h);
    let vein = saturate(vessels * density * 3.2);
    let vessel_color = mix(vec3<f32>(0.62, 0.27, 0.25), vec3<f32>(0.50, 0.09, 0.08), looks.x);
    albedo = mix(albedo, vessel_color, vein);
    // Diffuse hyperaemia with redness, following the vessel density.
    albedo = mix(albedo, albedo * vec3<f32>(1.0, 0.84, 0.82), saturate(looks.x * density * 1.2));
    // The conjunctiva is pinker toward the canthi, where it is thicker and more vascular.
    albedo *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.9, 0.87), smoothstep(0.35, 0.85, side));
    // Limbal darkening: the grey ring where the sclera meets the cornea.
    albedo *= 1.0 - looks.z * 0.45 * near_limbus;
    // The plica semilunaris (a pink fold) and the caruncle (moist pink tissue) at the medial
    // canthus.
    let fold = feature_noise(local * 30.0 + seed);
    let plica = (1.0 - smoothstep(0.28, 0.38, medial)) * smoothstep(0.14, 0.22, medial);
    albedo = mix(albedo, EYE_PLICA * (0.9 + 0.2 * fold), 0.55 * plica);
    let caruncle = 1.0 - smoothstep(0.12, 0.22, medial);
    albedo = mix(albedo, EYE_CARUNCLE * (0.8 + 0.4 * fold), caruncle);
    return max(albedo, vec3<f32>(0.0));
}

// The iris albedo at plane coordinates `local` (distance `radius` from the axis) [EYE-02].
fn eye_iris_albedo(slot: u32, s: EyeShape, local: vec2<f32>, radius: f32) -> vec3<f32> {
    let outer = feature_vec(slot, 5u);
    let inner = feature_vec(slot, 6u);
    let more = feature_vec(slot, 7u);
    let seed = more.w;
    // 0 at the pupil margin, 1 at the limbus: the layers stretch with the pupil.
    let t = saturate((radius - s.pupil) / max(s.limbus - s.pupil, 1e-6));
    let dir = local / max(radius, 1e-9);
    let wave = feature_noise(vec3<f32>(dir * 5.0, seed)) - 0.5;
    let collarette = outer.w + wave * 0.09;
    var albedo = mix(inner.rgb, outer.rgb, smoothstep(collarette - 0.03, collarette + 0.08, t));
    // Radial fibres: noise stretched along the radius.
    let fibre_a = feature_fbm(vec3<f32>(dir * 34.0, t * 2.5 + seed));
    let fibre_b = feature_noise(vec3<f32>(dir * 95.0, t * 5.0 + seed + 5.0));
    albedo *= 1.0 + inner.w * ((fibre_a - 0.5) * 1.3 + (fibre_b - 0.5) * 0.7);
    // The collarette ridge, lighter and raised.
    let ridge = exp(-sq((t - collarette) / 0.035));
    albedo *= 1.0 + 0.4 * ridge * (0.5 + fibre_b);
    // Crypts: dark lacunae around the collarette.
    let c = feature_noise(vec3<f32>(dir * 8.0, t * 3.5 + seed + 13.0));
    let crypt = smoothstep(0.6, 0.76, c) * smoothstep(0.06, 0.22, t) * (1.0 - smoothstep(0.7, 0.88, t));
    albedo *= 1.0 - more.x * crypt * 0.7;
    // Contraction furrows: faint concentric rings in the outer iris.
    albedo *= 1.0 - 0.15 * smoothstep(0.45, 0.6, t) * pow(0.5 + 0.5 * cos(t * 48.0 + wave * 6.0), 8.0);
    // The pupillary ruff: the dark pigment epithelium showing at the pupil margin.
    albedo *= mix(0.25, 1.0, smoothstep(0.0, 0.045, t));
    // The limbal ring.
    albedo *= 1.0 - more.z * smoothstep(1.0 - more.y, 1.0, t);
    return max(albedo, vec3<f32>(0.0));
}

// How much of a light refracted into the eye along `inside` (unit, toward the iris) reaches the
// iris point `iris`: the ray followed back to the limbus plane must pass inside the limbus (the
// opaque sclera shades the rim on the light's side), with a soft edge for the semi-clear limbus.
fn eye_iris_light(s: EyeShape, iris: vec3<f32>, inside: vec3<f32>) -> f32 {
    let up_along = -dot(inside, s.axis);
    if (up_along <= 1e-4) {
        return 0.0;
    }
    let rise = s.limbus_z - dot(iris - s.centre, s.axis);
    let at = iris - inside * (max(rise, 0.0) / up_along);
    let q = at - s.centre;
    let radial = length(q - s.axis * dot(q, s.axis));
    return 1.0 - smoothstep(s.limbus * 0.94, s.limbus * 1.04, radial);
}

// The mirror image of `light` in the tear film at `p` (normal `n`, toward the viewer `v`): the
// radiance of a rectangular light where the reflected view ray meets its panel, with an edge
// softened by the film's residual roughness (`soft`, a fraction of the panel); 0 for other kinds,
// which the GGX lobe covers. A near-mirror GGX with the sphere-light widening spreads a softbox's
// energy over a broad dim lobe; the real cornea shows the panel itself: the catchlight [EYE-04].
fn eye_mirror_light(light: Light, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, soft: f32) -> vec3<f32> {
    if (u32(light.color_kind.w) != LIGHT_RECT) {
        return vec3<f32>(0.0);
    }
    let center = light.position_range.xyz;
    let dir = light.direction_cos_outer.xyz;
    let refl = reflect(-v, n);
    let denom = dot(refl, dir);
    if (denom > -1e-4 || dot(p - center, dir) <= 0.0) {
        return vec3<f32>(0.0);
    }
    let t = dot(center - p, dir) / denom;
    let local = p + refl * t - center;
    let half_w = max(length(light.right_cos_inner.xyz), 1e-6);
    let half_h = max(length(light.up_size.xyz), 1e-6);
    let x = abs(dot(local, light.right_cos_inner.xyz)) / (half_w * half_w);
    let y = abs(dot(local, light.up_size.xyz)) / (half_h * half_h);
    let mask = (1.0 - smoothstep(1.0 - soft, 1.0 + soft, x)) * (1.0 - smoothstep(1.0 - soft, 1.0 + soft, y));
    return light.color_kind.rgb * mask;
}

// The tear film's normal on the sclera: the conjunctiva's fine relief (folds a few tenths of a
// millimetre across) tilts the smooth sphere normal `n` at point `p`.
fn eye_film_normal(s: EyeShape, p: vec3<f32>, n: vec3<f32>, strength: f32) -> vec3<f32> {
    let scale = 30.0 / max(s.radius, 1e-6);
    let e = 0.3 / scale;
    let n0 = feature_noise(p * scale);
    let g = vec3<f32>(
        feature_noise((p + vec3<f32>(e, 0.0, 0.0)) * scale) - n0,
        feature_noise((p + vec3<f32>(0.0, e, 0.0)) * scale) - n0,
        feature_noise((p + vec3<f32>(0.0, 0.0, e)) * scale) - n0,
    ) / 0.3;
    let tangential = g - n * dot(g, n);
    return safe_normalize(n - tangential * (0.018 * strength), n);
}

// The transmittance of the eye-occlusion film at lid openness `o` for lash density `lashes`:
// darkest at the margins, more under the upper lid, clear a third of the way into the opening.
fn eye_film(o: vec3<f32>, lashes: f32) -> f32 {
    let margin = 1.0 - smoothstep(0.0, 0.3, o.x);
    let upper = 1.0 - smoothstep(0.0, 0.45, o.y);
    return 1.0 - EYE_OCCLUSION_OPACITY * saturate(0.9 * margin + 0.8 * upper * (0.6 + 0.4 * lashes));
}

// The eyeball of entry `slot` seen along the camera ray through mesh point `world` (normal
// `ng`), under the occlusion film: the traced sclera, or the cornea and the iris behind it.
fn eye_ball(slot: u32, s: EyeShape, world: vec3<f32>, ng: vec3<f32>, camera: vec3<f32>, v: vec3<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let lashes = feature_float(slot, 62u);
    var t = eye_trace(s, camera, -v);
    if (t.kind == EYE_MISS) {
        // The mesh's silhouette reaches a little past the analytic eye: shade that as sclera.
        t.kind = EYE_SCLERA;
        t.point = world;
        t.normal = safe_normalize(world - s.centre, ng);
    }
    let o = eye_lid_openness(slot, s.centre, t.point);
    let lids = eye_lid_factors(o, lashes);
    let film = eye_film(o, lashes);
    let n = t.normal;
    let n_dot_v = max(dot(n, v), 1e-4);
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    let env = frame.environment.x;
    let tear = vec3<f32>(EYE_TEAR_F0);
    let looks = feature_vec(slot, 4u);
    var out: FeatureShade;
    out.normal = n;
    if (t.kind == EYE_SCLERA) {
        // The tear film: smooth, its normal following the conjunctiva's relief; the caruncle is
        // rougher tissue under the same film [EYE-03].
        let caruncle = 1.0 - smoothstep(0.12, 0.22, o.z);
        let wobble = feature_noise((t.point - s.centre) * (40.0 / max(s.radius, 1e-6))) - 0.5;
        let rough = clamp(looks.z * (1.0 + looks.w * 1.6 * wobble) + 0.12 * caruncle, 0.03, 1.0);
        let normal = eye_film_normal(s, t.point, n, 1.0 + looks.w + 2.0 * caruncle);
        out = feature_lit(t.point, normal, n, v, rough, EYE_TEAR_F0, mix(0.25, 0.45, caruncle), lids.y, lids.x, pixel, taps);
        out.albedo = eye_sclera_albedo(slot, s, t.point, o.z);
        for (var i = 0u; i < count; i++) {
            let light = frame.lights[i];
            let visibility = shadow_visibility(light, t.point, n, pixel, taps) * lids.y;
            out.specular += eye_mirror_light(light, t.point, normal, v, 0.15) * (f_dielectric(max(dot(normal, v), 1e-4), 1.337) * visibility * (1.0 - caruncle));
        }
        out.diffuse *= film;
        out.specular *= film;
        return out;
    }
    // The cornea: specular of the tear film, the iris (or the sclera at the limbus) behind it.
    let rough = EYE_CORNEA_ROUGHNESS;
    let alpha = alpha_from_roughness(rough);
    let dfg = dfg_lookup(n_dot_v, rough);
    let compensation = specular_energy_compensation(tear, dfg);
    let iris_n = s.axis;
    var specular = vec3<f32>(0.0);
    var diffuse = vec3<f32>(0.0);
    for (var i = 0u; i < count; i++) {
        let light = frame.lights[i];
        let visibility = shadow_visibility(light, t.point, n, pixel, taps) * lids.y;
        if (u32(light.color_kind.w) == LIGHT_RECT) {
            specular += eye_mirror_light(light, t.point, n, v, 0.08) * (f_dielectric(n_dot_v, 1.337) * visibility);
        } else {
            let sl = shade_light(light, t.point, n, v, alpha, tear, visibility);
            specular += sl.specular * compensation;
        }
        let incident = incident_light(light, t.point);
        let cos_in = dot(n, incident.l);
        if (cos_in > 0.0) {
            // Into the cornea, then onto the iris plane past the limbus; reduced by the Fresnel
            // transmission. The iris stroma scatters: a little wrap.
            let inside = refract(-incident.l, n, 1.0 / s.cornea_ior);
            let through = 1.0 - f_dielectric(cos_in, s.cornea_ior);
            let cos_iris = max((dot(-inside, iris_n) + 0.2) / 1.2, 0.0);
            let rim = select(1.0, eye_iris_light(s, t.iris, inside), t.radius <= s.limbus);
            diffuse += incident.irradiance * (visibility * through * cos_iris * rim * INV_PI);
        }
    }
    let view_through = 1.0 - f_dielectric(n_dot_v, s.cornea_ior);
    // The ambient light reaching the iris comes through the corneal dome: less toward the rim.
    let rim_ambient = mix(1.0, 0.7, smoothstep(0.6, 1.0, t.radius / max(s.limbus, 1e-6)));
    diffuse = (diffuse + sh_irradiance(iris_n) * (env * INV_PI * lids.x * rim_ambient)) * view_through;
    specular += env_prefiltered(reflect(-v, n), rough) * (env * mix(0.5, 1.0, lids.x) * (EYE_TEAR_F0 * dfg.x + dfg.y)) * compensation;
    if (t.radius < s.pupil) {
        out.albedo = EYE_PUPIL_ALBEDO;
    } else if (t.radius <= s.limbus) {
        out.albedo = eye_iris_albedo(slot, s, t.iris_local, t.radius);
    } else {
        // Past the limbus on the iris plane: the sclera seen through the corneal edge, the limbal
        // ring blending the two.
        let past = smoothstep(s.limbus, s.limbus * 1.08, t.radius);
        let edge = eye_iris_albedo(slot, s, t.iris_local, s.limbus);
        out.albedo = mix(edge, eye_sclera_albedo(slot, s, t.iris, o.z) * 0.85, past);
    }
    out.diffuse = diffuse * film;
    out.specular = specular * film;
    out.roughness = rough;
    return out;
}

// The eye's shading at a pixel of the eyeball (class EYE_CLASS), the tear line (TEAR_CLASS) or the
// occlusion shell (EYE_OCCLUSION_CLASS) of entry `entry` (or its partner): mesh point `world`
// with geometric normal `ng`, seen from `camera`.
fn eye_shade(entry: u32, part_class: u32, world: vec3<f32>, ng: vec3<f32>, camera: vec3<f32>, pixel: vec2<f32>, taps: u32) -> FeatureShade {
    let slot = eye_pick(entry, world);
    let s = eye_shape_of(slot);
    let v = safe_normalize(camera - world, ng);
    if (part_class != TEAR_CLASS) {
        // The occlusion film exists only over the eyeball: where the shell's rim reaches past it
        // (into the canthi) the ray behind it misses the eye and the lids or the caruncle behind
        // the shell must show.
        if (part_class == EYE_OCCLUSION_CLASS && eye_trace(s, camera, -v).kind == EYE_MISS) {
            discard;
        }
        // The eyeball, or the occlusion film over it.
        return eye_ball(slot, s, world, ng, camera, v, pixel, taps);
    }
    // The tear meniscus: its own sharp reflection over what shows through it.
    let lashes = feature_float(slot, 62u);
    let o = eye_lid_openness(slot, s.centre, world);
    let lids = eye_lid_factors(o, lashes);
    let behind = eye_trace(s, camera, -v);
    let local = (world - s.centre) / max(s.radius, 1e-6);
    if (behind.kind == EYE_MISS) {
        // Over the lid margin: the caruncle at the medial canthus, the wet lid margin elsewhere;
        // moist tissue under the tear film, its relief tilting the strip's flat normal.
        let caruncle = 1.0 - smoothstep(0.1, 0.3, o.z);
        let bumpy = eye_film_normal(s, world, ng, 10.0 + 14.0 * caruncle);
        var wet = feature_lit(world, bumpy, ng, v, mix(0.3, 0.22, caruncle), EYE_TEAR_F0, 0.45, max(lids.y, 0.4), max(lids.x, 0.5), pixel, taps);
        let fold = feature_noise(local * 28.0 + feature_float(slot, 31u));
        wet.albedo = mix(vec3<f32>(0.46, 0.25, 0.23), EYE_CARUNCLE * (0.75 + 0.45 * fold), caruncle);
        return wet;
    }
    var strip = feature_lit(world, ng, ng, v, 0.16, EYE_TEAR_F0, 0.3, max(lids.y, 0.5), max(lids.x, 0.6), pixel, taps);
    // The meniscus is a liquid mirror: the panels show in it as a thin bright line along the lid,
    // broken where the liquid thins between beads.
    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    let fresnel = f_dielectric(max(dot(ng, v), 1e-4), 1.337);
    let beads = smoothstep(0.35, 0.7, feature_noise(local * 18.0 + 5.0));
    for (var i = 0u; i < count; i++) {
        strip.specular += eye_mirror_light(frame.lights[i], world, ng, v, 0.2) * (fresnel * max(lids.y, 0.5) * 0.6 * beads);
    }
    var out = eye_ball(slot, s, world, ng, camera, v, pixel, taps);
    let through = 1.0 - f_dielectric(max(dot(ng, v), 1e-4), 1.337);
    out.diffuse *= through;
    out.specular = out.specular * through + strip.specular;
    return out;
}
