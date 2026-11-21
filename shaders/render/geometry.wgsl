// render/geometry.wgsl: the opaque geometry pass [RND-03] item 5. Vertex pulling from the scene
// streams; shades every light, the shadows and the image-based light, and writes
//   location 0 (rgba16float): diffuse lighting for unit albedo (irradiance / pi), weighted by
//              (1 - metalness): the input of the subsurface pass, multiplied by the albedo in the
//              composite; w = the view depth (metres) where the subsurface passes blur (skin
//              with SSS_MODE 0), else 0,
//   location 1 (rgba16float): specular radiance; w = the skin profile scale there, else 0,
//   location 2 (rgba32uint): the G-buffer-lite surface: x octahedral normal (pack_oct16), y sRGB
//              albedo + material class (4x8 unorm), z roughness, metalness, thickness, occlusion
//              (4x8 unorm), w the motion vector in pixels (pack2x16float) from the current and the
//              previous deformed positions [RND-04].
// Depth is reversed-Z (GreaterEqual), 32-bit float [RND-02].
//
// SHADOW_TAPS is the quality variant [QUAL-01]: PCF taps per shadowed light. SSS_MODE chooses the
// skin diffusion [SHD-02] [SHD-04]: 0 the separable screen-space pass after this one (Review,
// Still), 1 the pre-integrated falloff here (Interactive).
//
// Skin (material class 1) [SHD-01]: two GGX lobes (the material's roughness and the object's
// secondary roughness, mixed by the lobe mix) with f0 = the material's (0.028), multi-scatter
// compensation, and the diffuse weighted by 1 - the lobes' directional albedo so a white skin in a
// uniform environment reflects exactly what arrives (the furnace test). Light through thin parts
// (ears, nostrils) [SHD-03] comes from each shadowed light's map (render/subsurface.wgsl) and, where
// the compositor's inputs carry it, the baked thickness map [BAKE-01] as a lower bound.
//
// Surfaced skin (GpuObject ids.y bit OBJECT_SKIN_SURFACE, render/skin_surface.wgsl, group 3): the
// albedo, specular, roughness, cavity, scattering radius and zone masks come from the compositor's
// resolved set, the normal from its meso normal plus wrinkles [SKN-30] plus micro-detail [SKN-20],
// with the slope variance the filtering removed added to alpha^2; per light the pore-scale
// micro-shadowing [SHD-09], an oil / sweat coat (stronger and smoother on the lips) [SHD-06] and the
// vellus sheen lobe [SHD-07]; the environment light is occluded by the cavity (diffuse) and by
// the specular occlusion of Lagarde & de Rousiers 2014 [SHD-08], which also occludes every light's
// specular lobes and coat (todo 1.4: no glints in crevices the shadow map cannot resolve).
//
// Eyes and mouth (objects with an entry in the feature table, group 2 binding 7): shaded by
// eye/eye.wgsl (traced cornea, refracted iris, sclera, lid occlusion, tear line) [EYE-01]..[EYE-05]
// and mouth/mouth.wgsl (teeth, gums, tongue, cavity occlusion) [MTH-01]..[MTH-04].

#include "lib/brdf.wgsl"
#include "lib/color.wgsl"
#include "lib/packing.wgsl"
#include "render/frame.wgsl"
#include "render/scene.wgsl"
#include "render/lighting.wgsl"
#include "render/subsurface.wgsl"
#include "render/skin_surface.wgsl"
#include "eye/features.wgsl"
#include "eye/optics.wgsl"
#include "eye/eye.wgsl"
#include "mouth/mouth.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> prev_positions: array<f32>;
@group(1) @binding(2) var<storage, read> normals: array<f32>;
@group(1) @binding(3) var<storage, read> uvs: array<f32>;
@group(1) @binding(4) var<storage, read> objects: array<Object>;
@group(1) @binding(5) var<storage, read> indices: array<u32>;
@group(2) @binding(0) var env_specular: texture_cube<f32>;
@group(2) @binding(1) var env_sampler: sampler;
@group(2) @binding(2) var dfg_lut: texture_2d<f32>;
@group(2) @binding(3) var<storage, read> irradiance_sh: array<vec4<f32>, 9>;
@group(2) @binding(4) var shadow_atlas: texture_depth_2d_array;
@group(2) @binding(5) var shadow_sampler: sampler_comparison;
@group(2) @binding(6) var preintegrated_lut: texture_2d<f32>;
@group(2) @binding(7) var<storage, read> features: FeatureTable;
// Screen-space ambient occlusion (r) and contact shadows per shadow layer (render/occlusion.wgsl,
// todo 1.9); a white texel when that pass is off.
@group(2) @binding(8) var ambient_tex: texture_2d<f32>;
@group(2) @binding(9) var contact_tex: texture_2d<f32>;
@group(3) @binding(0) var skin_resolved: texture_2d_array<f32>;
@group(3) @binding(1) var skin_specular: texture_2d_array<f32>;
@group(3) @binding(2) var skin_inputs: texture_2d_array<f32>;
@group(3) @binding(3) var skin_detail: texture_2d<f32>;
@group(3) @binding(4) var wrinkle_maps: texture_2d_array<f32>;
@group(3) @binding(5) var<storage, read> wrinkle_activations: array<vec2<f32>>;
@group(3) @binding(6) var<storage, read> wrinkle_tile_slots: array<u32>;
@group(3) @binding(7) var clamp_sampler: sampler;
@group(3) @binding(8) var repeat_sampler: sampler;
@group(3) @binding(9) var<uniform> surface: SurfaceParams;

override SHADOW_TAPS: u32 = 16u;
override SSS_MODE: u32 = 0u;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) current: vec4<f32>,
    @location(4) previous: vec4<f32>,
    @location(5) @interpolate(flat) object: u32,
}

@vertex
fn vs_geometry(@builtin(vertex_index) vertex: u32, @builtin(instance_index) object_index: u32) -> VertexOut {
    let object = objects[object_index];
    let world = world_position(object, vertex);
    var out: VertexOut;
    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = world_normal(object, vertex);
    out.uv = fetch_uv(vertex);
    out.current = frame.unjittered_view_proj * vec4<f32>(world, 1.0);
    out.previous = frame.prev_view_proj * vec4<f32>(prev_world_position(object, vertex), 1.0);
    out.object = object_index;
    return out;
}

// The contact shadow of `light` at this pixel: its shadow layer's channel of the occlusion pass.
// A light without a shadow map (a fill, a large soft source) has nothing that closes the creases
// for it: the screen-space ambient occlusion stands in (1 without the occlusion pass).
fn contact_shadow(light: Light, contact: vec4<f32>, ambient: f32) -> f32 {
    if (light.shadow.x < 0.0) {
        return ambient;
    }
    return contact[min(u32(light.shadow.x), MAX_SHADOWED - 1u)];
}

struct GeometryOut {
    @location(0) diffuse: vec4<f32>,
    @location(1) specular: vec4<f32>,
    @location(2) surface: vec4<u32>,
}

@fragment
fn fs_geometry(in: VertexOut, @builtin(front_facing) front: bool) -> GeometryOut {
    let object = objects[in.object];
    var ng = safe_normalize(in.normal, vec3<f32>(0.0, 1.0, 0.0));
    if (!front) {
        ng = -ng;
    }
    // Derivatives in uniform control flow: the curvature for the pre-integrated falloff (normal
    // change per metre of surface) and the tangent frame of surfaced skin.
    let dpx = dpdx(in.world);
    let dpy = dpdy(in.world);
    let duvx = dpdx(in.uv);
    let duvy = dpdy(in.uv);
    let curvature = length(fwidth(ng)) / max(length(fwidth(in.world)), 1e-7);
    let tangents = surface_frame(ng, dpx, dpy, duvx, duvy);

    let feature = feature_slot(in.object);
    if (feature != 0u) {
        let part_class = u32(object.material.w + 0.5);
        let at = global_pixel(in.clip.xy);
        var fs: FeatureShade;
        if (feature_kind(feature) == FEATURE_EYE) {
            fs = eye_shade(feature, part_class, in.world, ng, frame.eye_exposure.xyz, at, SHADOW_TAPS);
        } else {
            fs = mouth_shade(feature, part_class, in.world, ng, frame.eye_exposure.xyz, at, SHADOW_TAPS);
        }
        var out: GeometryOut;
        out.diffuse = vec4<f32>(fs.diffuse, 0.0);
        out.specular = vec4<f32>(fs.specular, 0.0);
        let feature_albedo = vec4<f32>(linear_to_srgb_rgb(saturate(fs.albedo)), object.material.w / 255.0);
        let feature_params = vec4<f32>(fs.roughness, 0.0, 0.0, 1.0);
        let motion = (in.current.xy / in.current.w - in.previous.xy / in.previous.w) * vec2<f32>(0.5, -0.5) * frame.viewport.xy;
        out.surface = vec4<u32>(pack_oct16(fs.normal), pack4x8unorm(feature_albedo), pack4x8unorm(feature_params), pack2x16float(motion));
        return out;
    }

    // The occlusion targets may have half the resolution (below the Still tier).
    let occlusion_size = textureDimensions(ambient_tex);
    let screen = min(vec2<i32>(in.clip.xy * vec2<f32>(occlusion_size) * frame.viewport.zw), vec2<i32>(occlusion_size) - 1);
    let ambient_open = textureLoad(ambient_tex, screen, 0).r;
    let contact_open = textureLoad(contact_tex, screen, 0);
    let is_skin = u32(object.material.w + 0.5) == SKIN_CLASS;
    let material_rough = vec3<f32>(object.base_color_roughness.w, object.skin.x, object.skin.y);
    var surf = plain_surface(ng, max(object.base_color_roughness.rgb, vec3<f32>(0.0)), material_rough);
    if (is_skin && (object.ids.y & OBJECT_SKIN_SURFACE) != 0u) {
        let up = safe_normalize(object.model[1].xyz, vec3<f32>(0.0, 1.0, 0.0));
        let ahead = safe_normalize(object.model[2].xyz, vec3<f32>(0.0, 0.0, 1.0));
        surf = skin_surface(in.uv, ng, tangents, duvx, duvy, up, ahead, surf.albedo, material_rough);
    }
    let n = surf.normal;
    let v = safe_normalize(frame.eye_exposure.xyz - in.world, n);
    let n_dot_v = max(dot(n, v), 1e-4);
    let base = max(surf.albedo, vec3<f32>(0.0));
    // The slope variance removed by filtering widens the lobes: alpha^2 + variance (alpha = r^2).
    let roughness = sqrt(sqrt(pow(clamp(surf.roughness0, 0.02, 1.0), 4.0) + surf.alpha_widen));
    let metal = saturate(object.material.x);
    let f0 = mix(vec3<f32>(object.material.y * surf.f0_scale), base, metal);
    let alpha = alpha_from_roughness(roughness);
    let pixel = global_pixel(in.clip.xy);

    let count = min(u32(frame.environment.w), MAX_LIGHTS);
    let dfg = dfg_lookup(n_dot_v, roughness);
    let compensation = specular_energy_compensation(f0, dfg);
    let env = frame.environment.x;
    let reflected = reflect(-v, n);
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    var sss_scale = 0.0;
    if (is_skin) {
        let skin = object.skin;
        let roughness1 = sqrt(sqrt(pow(clamp(surf.roughness1, 0.02, 1.0), 4.0) + surf.alpha_widen));
        let alpha1 = alpha_from_roughness(roughness1);
        let blend = saturate(surf.lobe_mix);
        let scale = select(clamp(skin.z, 0.0, 4.0), surf.sss_scale, surf.sss_scale >= 0.0);
        let translucency = saturate(skin.w);
        // Profile scale / curvature radius in 1/mm.
        let inv_radius = curvature * 0.001 * scale;
        let dfg1 = dfg_lookup(n_dot_v, roughness1);
        let compensation1 = specular_energy_compensation(f0, dfg1);
        // The oil / sweat coat [SHD-06] attenuates what lies under it by its Fresnel reflectance.
        let coat = coat_strength(surf);
        let coat_rough = coat_roughness(surf);
        let coat_alpha = alpha_from_roughness(coat_rough);
        let coat_through = 1.0 - coat * f_schlick_scalar(COAT_F0, 1.0, n_dot_v);
        // Specular occlusion [SHD-08] (Lagarde & de Rousiers 2014) from the cavity: the baked ambient
        // occlusion times the pore cavity. It occludes the environment's specular and, as the
        // lobe's micro-occlusion, every light's: a crevice the shadow map is too coarse to resolve
        // (eye corners, nostrils, lip corners, pores) does not glint.
        // The baked occlusion and the screen-space one measure the same thing at different sizes:
        // the darker of the two, not their product (todo 1.9).
        let ao = min(surf.cavity, ambient_open) * surf.micro_cavity;
        let spec_occlusion = saturate(pow(n_dot_v + ao, exp2(-16.0 * alpha - 1.0)) - 1.0 + ao);
        for (var i = 0u; i < count; i++) {
            let light = frame.lights[i];
            var visibility = shadow_visibility(light, in.world, ng, pixel, SHADOW_TAPS) * contact_shadow(light, contact_open, ambient_open);
            let is_rect = u32(light.color_kind.w) == LIGHT_RECT;
            let incident = incident_light(light, in.world);
            let n_dot_l = dot(n, incident.l);
            if (!is_rect) {
                // Pore-scale micro-shadowing [SHD-09].
                visibility *= skin_micro_shadow(surf.micro_cavity, n_dot_l);
            }
            let s0 = shade_light(light, in.world, n, v, alpha, f0, visibility);
            let s1 = shade_light(light, in.world, n, v, alpha1, f0, visibility);
            specular += mix(s0.specular * compensation, s1.specular * compensation1, blend) * (coat_through * spec_occlusion);
            var d = s0.diffuse;
            if (!is_rect) {
                if (SSS_MODE == 1u) {
                    d = incident.irradiance * visibility * preintegrated_falloff(n_dot_l, inv_radius) * INV_PI;
                }
                if (translucency > 0.0 && scale > 0.0) {
                    let slab = shadow_slab(light, in.world, ng);
                    let shadow_mm = slab.x * 1000.0;
                    var thickness_mm = shadow_mm;
                    if (surf.thickness_mm >= 0.0 && shadow_mm >= 0.0) {
                        // The shadow map measures the tissue along this light's path; the baked
                        // thickness (along the inward normal, every part of the character) only
                        // bounds it from below where the map's resolution and bias underestimate
                        // it. Alone (the thinner of the two, or for lights without a shadow map) it
                        // let the far ear, the shadowed eyelids and the skin in front of the teeth
                        // glow red.
                        thickness_mm = max(shadow_mm, surf.thickness_mm);
                    }
                    // A point the light reaches directly is its own occluder in the shadow map: its
                    // "thickness" is only the THICKNESS_SHRINK offset (0.5 mm), whose transmittance
                    // is nearly the full light, so every lit surface near the terminator (the wrap
                    // region, n.l below 0.3) glowed orange: the ear rims and lid folds under the
                    // side light (todo 1.4). Light that crossed tissue comes from an occluder
                    // clearly behind the shrunk point; the transmission fades in from 1.5 to 3
                    // shrink distances.
                    let through = smoothstep(1.5 * THICKNESS_SHRINK_MM, 3.0 * THICKNESS_SHRINK_MM, shadow_mm);
                    if (thickness_mm >= 0.0 && through > 0.0) {
                        let wrap = saturate(TRANSLUCENCY_WRAP - dot(ng, incident.l));
                        // slab.y: the cosine at which the light entered the far side (todo 1.9).
                        d += incident.irradiance * skin_transmittance(thickness_mm, scale) * (wrap * translucency * through * slab.y * INV_PI);
                    }
                }
                if (n_dot_l > 0.0) {
                    let h = safe_normalize(incident.l + v, n);
                    let lit = incident.irradiance * (visibility * n_dot_l);
                    if (coat > 0.0) {
                        let fc = f_schlick_scalar(COAT_F0, 1.0, max(dot(v, h), 0.0));
                        let dv = coat_lobe(max(dot(n, h), 0.0), n_dot_v, n_dot_l, coat_alpha);
                        specular += lit * (coat * fc * dv * spec_occlusion);
                    }
                    if (surf.fuzz > 0.0) {
                        // Vellus sheen [SHD-07] [SKN-22]: strongest at grazing angles.
                        let sheen = d_charlie(surface.fuzz.y, max(dot(n, h), 0.0)) * v_neubelt(n_dot_v, n_dot_l);
                        specular += lit * (surf.fuzz * sheen);
                    }
                }
            }
            diffuse += d;
        }
        // Image-based light: both lobes, and the diffuse weighted by what the lobes do not reflect;
        // occluded by the cavity [SHD-08].
        let albedo0 = (f0 * dfg.x + dfg.y) * compensation;
        let albedo1 = (f0 * dfg1.x + dfg1.y) * compensation1;
        let spec_albedo = mix(albedo0, albedo1, blend);
        let env_spec = mix(env_prefiltered(reflected, roughness) * albedo0, env_prefiltered(reflected, roughness1) * albedo1, blend);
        specular += env_spec * (env * spec_occlusion * coat_through);
        if (coat > 0.0) {
            let coat_dfg = dfg_lookup(n_dot_v, coat_rough);
            specular += env_prefiltered(reflected, coat_rough) * (env * spec_occlusion * coat * (COAT_F0 * coat_dfg.x + coat_dfg.y));
        }
        diffuse = (diffuse + sh_irradiance(n) * (env * INV_PI * ao)) * (saturate(vec3<f32>(1.0) - spec_albedo) * coat_through);
        if (SSS_MODE == 0u) {
            sss_scale = scale;
        }
    } else {
        for (var i = 0u; i < count; i++) {
            let light = frame.lights[i];
            let visibility = shadow_visibility(light, in.world, n, pixel, SHADOW_TAPS) * contact_shadow(light, contact_open, ambient_open);
            let s = shade_light(light, in.world, n, v, alpha, f0, visibility);
            diffuse += s.diffuse;
            specular += s.specular;
        }
        diffuse += sh_irradiance(n) * (env * INV_PI * ambient_open);
        specular = specular * compensation
            + env_prefiltered(reflected, roughness) * (env * ambient_open) * (f0 * dfg.x + dfg.y) * compensation;
    }

    // Motion of this point from the previous frame to this one, in pixels (+x right, +y down).
    let current = in.current.xy / in.current.w;
    let previous = in.previous.xy / in.previous.w;
    let motion = (current - previous) * vec2<f32>(0.5, -0.5) * frame.viewport.xy;

    var out: GeometryOut;
    out.diffuse = vec4<f32>(diffuse * (1.0 - metal), select(0.0, in.current.w, sss_scale > 0.0));
    out.specular = vec4<f32>(specular, sss_scale);
    let albedo = vec4<f32>(linear_to_srgb_rgb(saturate(base)), object.material.w / 255.0);
    let params = vec4<f32>(roughness, metal, saturate(object.material.z), 1.0);
    out.surface = vec4<u32>(pack_oct16(n), pack4x8unorm(albedo), pack4x8unorm(params), pack2x16float(motion));
    return out;
}
