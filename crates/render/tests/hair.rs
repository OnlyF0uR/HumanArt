//! GPU tests of the hair renderer `[HR-01]`..`[HR-05]` on this machine's GPU (headless, one device
//! per binary, see `common`): the shader's fibre scattering against the CPU reference, and (below)
//! the rendering passes, the rubric scenes and the timings. Pictures go to
//! target/agents/hair-render/shots/.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

mod common;

use std::path::PathBuf;

use humanart_core::rng::Rng;
use humanart_gpu::resources::{Budget, BufferDesc, Category, Resources, Tag};
use humanart_render::hair::bsdf::{AzimuthalTable, DualTable, FibreDir, FibreParams, dual_scattering};
use humanart_render::hair::luts::HairLuts;

/// The folder of this work package's pictures.
fn shots() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/hair-render/shots");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A buffer of little-endian f32 words.
fn f32_words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Reads a whole buffer back as f32 words (test helper; blocking with the bounded wait).
fn read_f32(gpu: &humanart_gpu::Gpu, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hair test readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("the readback maps"));
    common::wait(&gpu.device);
    let words = {
        let bytes = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    };
    staging.unmap();
    words
}

/// `[HR-02]` the shader's dual-scattered fibre response (tables on the GPU in f16, filtered by
/// the sampler) matches the CPU reference for random fibres, directions, absorptions and
/// depths into the hair.
#[test]
fn hr_02_gpu_fibre_scattering_matches_the_cpu_reference() {
    let gpu = common::gpu();
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let params = FibreParams::default();
    let luts = HairLuts::new(&mut resources, &params).unwrap();
    let table = AzimuthalTable::build(&params).unwrap();
    let dual = DualTable::build(&table);
    let mut rng = Rng::new(5);
    let unit = |rng: &mut Rng| {
        let z = 1.0 - 2.0 * rng.f32();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = std::f32::consts::TAU * rng.f32();
        [z, r * a.cos(), r * a.sin()]
    };
    let count = 2048;
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..count {
        // A random fibre frame: tangent t, and two vectors in it.
        let t = unit(&mut rng);
        let wo = unit(&mut rng);
        let wi = unit(&mut rng);
        let sigma = rng.range_f32(0.0, 6.0) * rng.f32();
        let tau = if rng.f32() < 0.3 { 0.0 } else { rng.range_f32(0.0, 6.0) };
        cases.extend_from_slice(&[sigma, sigma, sigma, tau]);
        cases.extend_from_slice(&[t[0], t[1], t[2], 0.0]);
        cases.extend_from_slice(&[wo[0], wo[1], wo[2], 0.0]);
        cases.extend_from_slice(&[wi[0], wi[1], wi[2], 0.0]);
        // The fibre frame of the CPU reference: x along t, y and z across it.
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let helper = if t[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let d = dot(helper, t);
        let mut y = [helper[0] - d * t[0], helper[1] - d * t[1], helper[2] - d * t[2]];
        let ly = dot(y, y).sqrt();
        y = y.map(|c| c / ly);
        let z = [t[1] * y[2] - t[2] * y[1], t[2] * y[0] - t[0] * y[2], t[0] * y[1] - t[1] * y[0]];
        let local = |w: [f32; 3]| FibreDir::from_vector([dot(w, t), dot(w, y), dot(w, z)].map(f64::from));
        expected.push(dual_scattering(&table, &dual, sigma, tau, local(wo), local(wi)));
    }
    let make = |resources: &mut Resources, bytes: &[u8], usage, purpose| {
        let (id, _) = resources
            .create_buffer(&BufferDesc {
                size: bytes.len().max(16) as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::shared(Category::Other, purpose),
            })
            .unwrap();
        resources.write_buffer(id, 0, bytes).unwrap();
        resources.buffer(id).unwrap().raw().clone()
    };
    let v = params.variances();
    let fibre = f32_words(&[v[0], v[1], v[2], v[3], params.shift(0), params.shift(1), params.shift(2), 0.0]);
    let fibre = make(&mut resources, &fibre, wgpu::BufferUsages::UNIFORM, "probe fibre");
    let cases = make(&mut resources, &f32_words(&cases), wgpu::BufferUsages::STORAGE, "probe cases");
    let out = make(&mut resources, &vec![0; count * 16], wgpu::BufferUsages::STORAGE, "probe out");
    let source = humanart_shaders::compose("hair/probe.wgsl", &[]).unwrap().into_source();
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hair probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hair probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hair probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(luts.azimuthal_view()) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(luts.dual_view()) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(luts.sampler()) },
            wgpu::BindGroupEntry { binding: 3, resource: fibre.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: cases.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let got = read_f32(gpu, &out);
    let peak = expected.iter().copied().fold(0.0_f32, f32::max);
    let mut worst = (0.0_f32, 0);
    let mut sum_abs = 0.0_f64;
    for (i, &e) in expected.iter().enumerate() {
        let g = got[i * 4];
        assert!(g.is_finite(), "case {i}: {g}");
        // Every channel saw the same sigma: the channels agree.
        assert!((got[i * 4 + 1] - g).abs() <= 1e-4 * (1.0 + g.abs()), "case {i}: channels differ");
        // Relative to the case's own value, with a floor at 1 % of the peak.
        let err = (g - e).abs() / e.abs().max(0.01 * peak);
        sum_abs += f64::from((g - e).abs());
        if err > worst.0 {
            worst = (err, i);
        }
    }
    println!(
        "[HR-02] GPU vs CPU fibre response: {count} cases, peak {peak:.3}, worst relative {:.4} (case {}), mean abs {:.2e}",
        worst.0,
        worst.1,
        sum_abs / count as f64
    );
    assert!(worst.0 < 0.01, "worst relative error {} at case {}", worst.0, worst.1);
    let _ = shots();
}

/// `[HR-01]` smoke test of the hair passes in the core renderer: 4000 dark-brown strands hang in
/// front of a grey card under a key light; the frame renders without a validation error, the
/// visibility buffer has coverage, the hair pixels are finite and darker than the bare card, and
/// the picture goes to target/agents/hair-render/shots/hair_smoke.png.
#[test]
fn hr_01_hair_passes_render_a_strand_patch() {
    use humanart_math::Vec3;
    use humanart_render::camera::{Camera, Exposure};
    use humanart_render::hair::{HairGroom, HairRenderer, HairSettings};
    use humanart_render::light::{Background, Light, LightRig, ProceduralEnvironment};
    use humanart_render::renderer::RenderConfig;
    use humanart_render::scene::{Material, Mesh, Scene};

    let (w, h) = (256_u32, 256_u32);
    let card = [
        Vec3::new(-0.3, -0.3, -0.1),
        Vec3::new(0.3, -0.3, -0.1),
        Vec3::new(0.3, 0.3, -0.1),
        Vec3::new(-0.3, 0.3, -0.1),
    ];
    let mesh = Mesh::from_polygons(&card, [&[0_u32, 1, 2, 3][..]]).unwrap();
    let mut scene = Scene::default();
    scene.add(mesh, Material { base_color: [0.5; 3], roughness: 0.8, f0: 0.02, ..Material::default() });
    let config = RenderConfig { taa: false, ..RenderConfig::new(w, h) };
    let mut harness = common::Harness::new(config, &scene);
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 0.6),
        target: Vec3::ZERO,
        focal_mm: 50.0,
        near: 0.05,
        exposure: Exposure::ev100(0.0),
        ..Camera::default()
    };
    let rig = LightRig {
        name: "hair smoke",
        lights: vec![Light::directional(Vec3::new(-0.4, -0.5, -1.0), 3.0, 5600.0)],
        environment: ProceduralEnvironment::Uniform,
        environment_nits: 0.3,
        environment_rotation: 0.0,
        background: Background::Gray(0.18),
        ev100: 0.0,
    };
    harness.frames(&camera, &rig, 1);
    let bare = harness.resolved();

    // The strands: a 60 x 60 grid of 16-point vertical strands with a little sway.
    let (strands, points) = (3600_u32, 16_u32);
    let mut rng = Rng::new(3);
    let mut pts = Vec::new();
    let mut attrs = Vec::new();
    for s in 0..strands {
        let (i, j) = (s % 60, s / 60);
        let x = -0.1 + 0.2 * (i as f32 + rng.f32()) / 60.0;
        let z = -0.02 + 0.04 * (j as f32 + rng.f32()) / 60.0;
        let phase = rng.f32() * std::f32::consts::TAU;
        for k in 0..points {
            let t = k as f32 / (points - 1) as f32;
            pts.extend_from_slice(&[x + 0.004 * (phase + 8.0 * t).sin(), 0.12 - 0.24 * t, z, 7e-5]);
        }
        let color_id = (rng.f32() * 65_535.0) as u32;
        attrs.extend_from_slice(&[0_u32, 0, 0, 0, color_id, 0, 0, 0]);
    }
    let make = |resources: &mut Resources, bytes: Vec<u8>, purpose| {
        let (id, _) = resources
            .create_buffer(&BufferDesc {
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::shared(Category::Other, purpose),
            })
            .unwrap();
        resources.write_buffer(id, 0, &bytes).unwrap();
        resources.buffer(id).unwrap().raw().clone()
    };
    let points_buffer = make(&mut harness.resources, f32_words(&pts), "smoke points");
    let attr_bytes: Vec<u8> = attrs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let attrs_buffer = make(&mut harness.resources, attr_bytes, "smoke attrs");
    let color = humanart_model::hair::HairColor { eumelanin: 1.3, pheomelanin: 0.2, ..Default::default() };
    let groom = HairGroom {
        points: points_buffer,
        attrs: attrs_buffer,
        strands,
        points_per_strand: points,
        color,
        centre: Vec3::ZERO,
        radius: 0.2,
        diameter: 7e-5,
        length: 0.24,
        min_keep: humanart_render::hair::view::LOD_MIN_KEEP,
    };
    let mut hair =
        HairRenderer::new(harness.gpu, &mut harness.resources, HairSettings::default(), &FibreParams::default())
            .unwrap();
    hair.set_grooms(&mut harness.resources, &[groom]).unwrap();
    harness.renderer.set_hair(&mut harness.resources, Some(hair)).unwrap();
    harness.frames(&camera, &rig, 1);
    let with_hair = harness.resolved();
    harness.save_display("hair_smoke.png");
    let png = common::shots_dir().join("hair_smoke.png");
    let _ = std::fs::copy(&png, shots().join("hair_smoke.png"));

    let hair_renderer = harness.renderer.hair().unwrap();
    assert_eq!(hair_renderer.dom_count(), 1);
    let [_, coverage, _] = hair_renderer.visibility_buffers().unwrap();
    let coverage = read_f32(harness.gpu, coverage);
    let covered = coverage.iter().filter(|c| c.to_bits() > 0).count();
    // The centre pixel lies inside the patch.
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: [f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    println!(
        "[HR-01] covered pixels {covered}, centre luminance bare {:.4} with hair {:.4}",
        lum(bare[centre]),
        lum(with_hair[centre])
    );
    assert!(covered > 1000, "{covered} pixels covered");
    assert!(with_hair.iter().all(|p| p.iter().all(|c| c.is_finite())));
    assert!(lum(with_hair[centre]) < lum(bare[centre]), "dark hair over a grey card darkens it");
}

// ---------------------------------------------------------------------------------------------
// Step 5 (todo 1.7): the groom set on the generated template's head.

mod template_head {
    use std::fmt::Write as _;
    use std::sync::OnceLock;

    use humanart_core::rng::Rng;
    use humanart_model::Template;
    use humanart_model::generate::build::{Options, template};
    use humanart_model::hair::{
        CharacterGroom, GroomKind, GroomLayout, GroomSetSpec, ScalpStyle, TemplateHair, character_grooms,
    };
    use humanart_render::camera::{Camera, Exposure, ReviewCamera};
    use humanart_render::groom::{GpuGroom, GroomCapacity};
    use humanart_render::hair::bsdf::FibreParams;
    use humanart_render::hair::{HairGroom, HairRenderer, HairSettings};
    use humanart_render::ibl::EnvironmentSource;
    use humanart_render::light::{BuiltinRig, LightRig};
    use humanart_render::renderer::RenderConfig;
    use humanart_render::scene::{Material, Mesh, Scene};

    use super::common;

    /// The generated template and its hair surface, built once per test binary.
    pub fn fixture() -> &'static (Template, TemplateHair) {
        static F: OnceLock<(Template, TemplateHair)> = OnceLock::new();
        F.get_or_init(|| {
            // The generator runs for minutes unoptimised: build it on a thread and mark progress
            // for the watchdog while it works (it still catches a generator that never ends).
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let t = template(&Options { lods: false, ..Options::default() }).unwrap().template;
                let hair = TemplateHair::rest(&t).unwrap();
                let _ = tx.send((t, hair));
            });
            let start = std::time::Instant::now();
            loop {
                match rx.recv_timeout(std::time::Duration::from_secs(5)) {
                    Ok(f) => break f,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        assert!(start.elapsed().as_secs() < 1800, "the template took over 30 min");
                        common::progress();
                    }
                    Err(e) => panic!("the template build failed: {e}"),
                }
            }
        })
    }

    /// The review folder.
    pub fn review() -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/review/hair");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The template as a scene: the skin part in a skin material, the other parts light grey.
    fn scene(t: &Template) -> Scene {
        let mut scene = Scene::default();
        let skin = Material { base_color: [0.50, 0.33, 0.25], roughness: 0.5, f0: 0.028, ..Material::default() };
        let other = Material { base_color: [0.6, 0.58, 0.56], roughness: 0.3, f0: 0.03, ..Material::default() };
        for part in t.mesh().parts() {
            let mesh = Mesh::from_template(t.mesh(), Some(&part.name)).unwrap();
            scene.add(mesh, if part.name == "body" { skin } else { other });
        }
        scene
    }

    /// The groom set of `style` with `strands` scalp strands (brows and lashes, no fuzz).
    pub fn grooms(style: ScalpStyle, strands: u32) -> Vec<CharacterGroom> {
        let (_, hair) = fixture();
        let spec = GroomSetSpec { scalp: style, scalp_strands: strands, peach_fuzz: false, ..GroomSetSpec::default() };
        character_grooms(hair, &spec).unwrap()
    }

    /// `layout` with its strands in the order `perm` (new strand `i` is old strand `perm[i]`);
    /// clump centres follow their strands.
    pub fn permuted(layout: &GroomLayout, perm: &[usize]) -> GroomLayout {
        let mut inverse = vec![0_u32; perm.len()];
        for (new, &old) in perm.iter().enumerate() {
            inverse[old] = new as u32;
        }
        GroomLayout {
            triangle: perm.iter().map(|&o| layout.triangle[o]).collect(),
            bary: perm.iter().map(|&o| layout.bary[o]).collect(),
            guides: perm.iter().map(|&o| layout.guides[o]).collect(),
            weights: perm.iter().map(|&o| layout.weights[o]).collect(),
            clump: perm.iter().map(|&o| inverse[layout.clump[o] as usize]).collect(),
            key: perm.iter().map(|&o| layout.key[o]).collect(),
            factors: perm.iter().map(|&o| layout.factors[o]).collect(),
            direction: perm.iter().map(|&o| layout.direction[o]).collect(),
            color: perm.iter().map(|&o| layout.color[o]).collect(),
            hairline: perm.iter().map(|&o| layout.hairline[o]).collect(),
        }
    }

    /// A random permutation of `n` strands.
    pub fn shuffle(n: usize, seed: u64) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(seed);
        for i in (1..n).rev() {
            perm.swap(i, rng.index(i + 1));
        }
        perm
    }

    /// A harness on the template scene.
    pub fn harness(config: RenderConfig) -> common::Harness {
        common::Harness::new(config, &scene(&fixture().0))
    }

    /// Uploads, generates and attaches (rest skin) every groom of `set`, and sets them as `h`'s
    /// hair with `settings`. Returns the GPU grooms (keep them alive while drawing).
    pub fn install(h: &mut common::Harness, set: &[CharacterGroom], settings: HairSettings) -> Vec<GpuGroom> {
        let (_, hair) = fixture();
        let (head, radius) = hair.head();
        let mut gpu_grooms = Vec::new();
        let mut drawn = Vec::new();
        let mut encoder =
            h.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grooms") });
        for g in set {
            let points = g.params.point_count();
            let capacity = GroomCapacity { strands: g.layout.len() as u32, guides: g.guides.len() as u32, points };
            let sdf = (g.kind == GroomKind::Scalp).then(|| hair.sdf());
            let mut gg = GpuGroom::new(h.gpu, &mut h.resources, hair.surface(), sdf, capacity).unwrap();
            gg.set_params(&mut h.resources, &g.params).unwrap();
            gg.set_world_guides(&mut h.resources, &g.guides.resampled_world(points, hair.surface()).unwrap()).unwrap();
            gg.set_layout(&mut h.resources, hair.surface(), &g.layout).unwrap();
            gg.bind_rest_skin(&h.gpu.device).unwrap();
            assert!(gg.encode_generate(&mut encoder) && gg.encode_attach(&mut encoder));
            let drawn_groom = HairGroom::from_gpu(&gg, &g.params, head, radius + 1.1 * g.length, g.length);
            // Sparse grooms are never thinned (their survivors would show as specks).
            drawn.push(if g.kind == GroomKind::Scalp { drawn_groom } else { drawn_groom.with_min_keep(1.0) });
            gpu_grooms.push(gg);
        }
        h.gpu.queue.submit([encoder.finish()]);
        if let Some(r) = h.renderer.hair_mut() {
            r.set_settings(&mut h.resources, settings).unwrap();
            r.set_grooms(&mut h.resources, &drawn).unwrap();
        } else {
            let mut r = HairRenderer::new(h.gpu, &mut h.resources, settings, &FibreParams::default()).unwrap();
            r.set_grooms(&mut h.resources, &drawn).unwrap();
            h.renderer.set_hair(&mut h.resources, Some(r)).unwrap();
        }
        gpu_grooms
    }

    /// The rig `kind` around the head, its environment set on `h`'s renderer.
    pub fn rig(h: &mut common::Harness, kind: BuiltinRig) -> LightRig {
        let (_, hair) = fixture();
        let (head, radius) = hair.head();
        let r = LightRig::builtin(kind, head, radius);
        h.renderer.set_environment(&mut h.resources, EnvironmentSource::Procedural(r.environment)).unwrap();
        r
    }

    /// The review camera `kind` around the head (`scale` times its sphere), with the rig's
    /// exposure.
    pub fn camera(kind: ReviewCamera, scale: f32, aspect: f32, rig: &LightRig) -> Camera {
        let (_, hair) = fixture();
        let (head, radius) = hair.head();
        let c = Camera::review(kind, head, radius * scale, aspect);
        Camera { exposure: Exposure::ev100(rig.ev100), ..c }
    }

    /// Hair pixels: where the visibility buffer has coverage.
    pub fn hair_mask(h: &common::Harness) -> Vec<bool> {
        let [_, coverage, _] = h.renderer.hair().unwrap().visibility_buffers().unwrap();
        super::read_f32(h.gpu, coverage).iter().map(|c| c.to_bits() > 0).collect()
    }

    /// Rec.709 luminance.
    pub fn lum(p: [f32; 4]) -> f32 {
        0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
    }

    /// Over the pixels of `mask`: mean and largest absolute luminance difference of `a` and `b`,
    /// and the mean luminance of `a`.
    pub fn compare(a: &[[f32; 4]], b: &[[f32; 4]], mask: &[bool]) -> (f32, f32, f32) {
        let (mut sum, mut worst, mut base, mut n) = (0.0_f64, 0.0_f32, 0.0_f64, 0_u32);
        for ((pa, pb), &m) in a.iter().zip(b).zip(mask) {
            if m {
                let d = (lum(*pa) - lum(*pb)).abs();
                sum += f64::from(d);
                base += f64::from(lum(*pa));
                worst = worst.max(d);
                n += 1;
            }
        }
        let n = f64::from(n.max(1));
        ((sum / n) as f32, worst, (base / n) as f32)
    }

    /// The median GPU times over `frames` frames (after 8 warm-up frames) of `cam`: the hair
    /// passes, each hair scope, and the whole frame, in milliseconds.
    pub fn hair_ms(
        h: &mut common::Harness,
        cam: &humanart_render::camera::Camera,
        rig: &LightRig,
        frames: u32,
    ) -> (f64, String, f64) {
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let (mut hair, mut total) = (Vec::new(), Vec::new());
        let mut scopes: std::collections::BTreeMap<&'static str, Vec<f64>> = std::collections::BTreeMap::new();
        for i in 0..8 + frames {
            h.frame(cam, rig, &[], false, None, i == 0);
            common::wait(&h.gpu.device);
            if h.renderer.collect_timings() && i >= 8 {
                let report = h.renderer.timings();
                let mut sum = 0.0;
                for s in report.scopes.iter().filter(|s| s.name.starts_with("hair")) {
                    let ms = s.gpu_nanos.unwrap_or(0) as f64 * 1e-6;
                    sum += ms;
                    scopes.entry(s.name).or_default().push(ms);
                }
                hair.push(sum);
                total.push(report.gpu_total_nanos().unwrap_or(0) as f64 * 1e-6);
            }
        }
        let mut list = String::new();
        for (name, mut v) in scopes {
            let _ = write!(list, "{name} {:.3}; ", median(&mut v));
        }
        (median(&mut hair), list, median(&mut total))
    }

    /// The prefix of this round's stills (earlier rounds' stills stay for comparison).
    pub const ROUND: &str = "r4-";

    /// Saves `h`'s display as target/review/hair/[`ROUND`]`name` (and in the shots folder).
    pub fn save(h: &mut common::Harness, name: &str) -> Vec<u8> {
        let name = &format!("{ROUND}{name}");
        let (w, ht, rgba) = h.display();
        let png = humanart_render::capture::encode_png_rgba8(w, ht, &rgba).unwrap();
        std::fs::write(review().join(name), &png).unwrap();
        let _ = std::fs::write(super::shots().join(name), &png);
        rgba
    }
}

/// `[HR-01]` `[HR-03]` `[HR-05]` step 5 on the generated template's head with the groom set of
/// todo 1.7 (medium straight 100k strands, brows, lashes): the review stills (back, profile,
/// harsh side light, back light, front; long wavy back) in target/review/hair/; TAA stability
/// (frame-to-frame change of the converged image on hair pixels); order invariance (the same
/// strands in a shuffled order); the ribbon fallback run and compared with the compute raster;
/// GPU timings at 1440p for the close-up and the far field. One test so the GPU is not shared
/// while timing.
#[test]
fn hr_05_template_head_scenes_stability_order_ribbon_and_timings() {
    use humanart_model::hair::{GroomKind, ScalpStyle};
    use humanart_render::camera::ReviewCamera;
    use humanart_render::hair::{HairPath, HairSettings};
    use humanart_render::light::BuiltinRig;
    use humanart_render::renderer::RenderConfig;
    use template_head::{camera, compare, hair_mask, harness, install, lum, rig, save};

    let set = template_head::grooms(ScalpStyle::MediumStraight, 100_000);
    let strands: usize = set.iter().map(|g| g.layout.len()).sum();
    println!("groom set: {strands} strands in {} grooms", set.len());

    // 1. Review stills, TAA on, 32 frames each.
    let (w, h) = (720_u32, 900_u32);
    let aspect = w as f32 / h as f32;
    let mut hx = harness(RenderConfig::new(w, h));
    let keep = install(&mut hx, &set, HairSettings::default());
    for (name, kind, rig_kind, scale) in [
        ("back.png", ReviewCamera::HairBack, BuiltinRig::NeutralStudio, 1.0),
        ("profile.png", ReviewCamera::ProfileLeft, BuiltinRig::NeutralStudio, 1.0),
        ("side_light.png", ReviewCamera::ThreeQuarterLeft, BuiltinRig::HarshSidelight, 1.0),
        ("back_light.png", ReviewCamera::ThreeQuarterRight, BuiltinRig::Backlight, 1.0),
        ("front.png", ReviewCamera::Front, BuiltinRig::NeutralStudio, 1.0),
    ] {
        let r = rig(&mut hx, rig_kind);
        let cam = camera(kind, scale, aspect, &r);
        hx.frames(&cam, &r, 32);
        save(&mut hx, name);
    }

    // 2. TAA stability: a converged profile view; the mean change from one frame to the next on
    //    hair pixels, against the same measure on the rest of the image.
    let r = rig(&mut hx, BuiltinRig::NeutralStudio);
    let cam = camera(ReviewCamera::ProfileLeft, 1.0, aspect, &r);
    hx.frames(&cam, &r, 40);
    let mask = hair_mask(&hx);
    let rest: Vec<bool> = mask.iter().map(|m| !m).collect();
    let mut prev = hx.resolved();
    let (mut hair_change, mut rest_change, mut hair_level) = (0.0_f32, 0.0_f32, 0.0_f32);
    for _ in 0..8 {
        hx.frame(&cam, &r, &[], false, None, false);
        let now = hx.resolved();
        let (dh, _, base) = compare(&prev, &now, &mask);
        let (dr, _, _) = compare(&prev, &now, &rest);
        hair_change += dh / 8.0;
        rest_change += dr / 8.0;
        hair_level = base;
        prev = now;
    }
    let covered = mask.iter().filter(|m| **m).count();
    println!(
        "[HR-05] TAA stability (profile, converged, 8 frames): {covered} hair pixels, mean frame-to-frame \
         luminance change {hair_change:.5} on hair ({:.2} % of its mean {hair_level:.4}), {rest_change:.5} elsewhere",
        100.0 * hair_change / hair_level.max(1e-6)
    );
    assert!(covered > 20_000, "{covered}");
    assert!(hair_change < 0.05 * hair_level, "hair flickers: {hair_change} of {hair_level}");
    drop(keep);

    // 3. Order invariance and 4. the ribbon fallback: TAA off (no jitter), full detail (no hash
    //    thinning, so every strand is drawn in both orders).
    let config = RenderConfig { taa: false, ..RenderConfig::new(w, h) };
    let full = HairSettings { full_detail: true, ..HairSettings::default() };
    let mut hy = harness(config);
    let r = rig(&mut hy, BuiltinRig::HarshSidelight);
    let cam = camera(ReviewCamera::ThreeQuarterLeft, 1.0, aspect, &r);
    let keep = install(&mut hy, &set, full);
    hy.frames(&cam, &r, 2);
    let reference = hy.resolved();
    let mask = hair_mask(&hy);
    save(&mut hy, "order_reference.png");
    let shuffled: Vec<_> = set
        .iter()
        .map(|g| {
            let perm = template_head::shuffle(g.layout.len(), 77);
            humanart_model::hair::CharacterGroom { layout: template_head::permuted(&g.layout, &perm), ..g.clone() }
        })
        .collect();
    let keep2 = install(&mut hy, &shuffled, full);
    hy.frames(&cam, &r, 2);
    let reordered = hy.resolved();
    let (mean, worst, base) = compare(&reference, &reordered, &mask);
    let differing = reference.iter().zip(&reordered).filter(|(a, b)| (lum(**a) - lum(**b)).abs() > 1e-3).count();
    println!(
        "[HR-01] order invariance (100k strands shuffled): mean |dL| {mean:.2e} on hair (mean L {base:.4}), \
         worst {worst:.4}, {differing} pixels differ by more than 1e-3"
    );
    assert!(mean < 0.01 * base, "the strand order changes the image: {mean} of {base}");
    drop(keep2);

    // The split walks draw the same fragments as one thread per segment.
    let keep_split = install(&mut hy, &set, HairSettings { split_walks: false, ..full });
    hy.frames(&cam, &r, 2);
    let unsplit = hy.resolved();
    let (mean, worst, base) = compare(&reference, &unsplit, &mask);
    println!(
        "[HR-04] split walks vs one thread per segment: mean |dL| {mean:.2e} (mean L {base:.4}), worst {worst:.4}"
    );
    assert!(mean < 0.001 * base, "splitting the walks changes the image: {mean} of {base}");
    drop(keep_split);

    // Skipping the segments hidden behind the scene draws the same fragments in front of it.
    let keep_cull = install(&mut hy, &set, HairSettings { cull_hidden: false, ..full });
    hy.frames(&cam, &r, 2);
    let unculled = hy.resolved();
    let (mean, worst, base) = compare(&reference, &unculled, &mask);
    println!("[HR-04] hidden segments culled vs walked: mean |dL| {mean:.2e} (mean L {base:.4}), worst {worst:.4}");
    assert!(mean < 0.001 * base, "culling hidden segments changes the image: {mean} of {base}");
    drop(keep_cull);

    let keep3 = install(&mut hy, &set, HairSettings { path: HairPath::Ribbon, ..full });
    hy.frames(&cam, &r, 2);
    let ribbon = hy.resolved();
    let ribbon_mask = hair_mask(&hy);
    save(&mut hy, "ribbon.png");
    let (mean, worst, base) = compare(&reference, &ribbon, &mask);
    let union = mask.iter().zip(&ribbon_mask).filter(|(a, b)| **a || **b).count();
    let both = mask.iter().zip(&ribbon_mask).filter(|(a, b)| **a && **b).count();
    println!(
        "[HR-01] ribbon fallback vs compute raster: mean |dL| {mean:.4} on hair (mean L {base:.4}, {:.1} %), worst \
         {worst:.3}; coverage IoU {:.3} ({} vs {} covered pixels)",
        100.0 * mean / base.max(1e-6),
        both as f32 / union.max(1) as f32,
        ribbon_mask.iter().filter(|m| **m).count(),
        mask.iter().filter(|m| **m).count()
    );
    assert!(ribbon.iter().all(|p| p.iter().all(|c| c.is_finite())));
    assert!(both as f32 > 0.5 * union as f32, "the ribbons cover the same hair");
    drop(keep3);
    drop(keep);
    drop(hy);

    // Peach fuzz on the cheek in back light, close: a fine down, not specks.
    let (_, th) = template_head::fixture();
    let fuzz_spec = humanart_model::hair::GroomSetSpec {
        scalp: ScalpStyle::MediumStraight,
        scalp_strands: 100_000,
        peach_fuzz: true,
        ..humanart_model::hair::GroomSetSpec::default()
    };
    let fuzz_set = humanart_model::hair::character_grooms(th, &fuzz_spec).unwrap();
    let fuzz_strands =
        fuzz_set.iter().filter(|g| g.kind == GroomKind::PeachFuzz).map(|g| g.layout.len()).sum::<usize>();
    let mut hf = harness(RenderConfig::new(w, h));
    let keep = install(&mut hf, &fuzz_set, HairSettings::default());
    let aspect = w as f32 / h as f32;
    let r = rig(&mut hf, BuiltinRig::Backlight);
    let mut cam = camera(ReviewCamera::ProfileLeft, 0.45, aspect, &r);
    // The left cheek: beside and above the mouth (the character's left is +X).
    let cheek = th.anchors().mouth + humanart_math::Vec3::new(0.045, 0.03, -0.015);
    let offset = cam.position - cam.target;
    cam.target = cheek;
    cam.position = cheek + offset;
    hf.frames(&cam, &r, 32);
    save(&mut hf, "peach_fuzz_back_light.png");
    println!("[HAIR-04] peach fuzz: {fuzz_strands} strands, never thinned; still peach_fuzz_back_light.png");
    drop(keep);

    // The long wavy style from behind.
    let wavy = template_head::grooms(ScalpStyle::LongWavy, 100_000);
    let mut hw = harness(RenderConfig::new(w, h));
    let keep = install(&mut hw, &wavy, HairSettings::default());
    let aspect = w as f32 / h as f32;
    let r = rig(&mut hw, BuiltinRig::NeutralStudio);
    hw.frames(&camera(ReviewCamera::HairBack, 1.6, aspect, &r), &r, 32);
    save(&mut hw, "long_wavy_back.png");
    let r = rig(&mut hw, BuiltinRig::Backlight);
    hw.frames(&camera(ReviewCamera::ProfileLeft, 1.6, aspect, &r), &r, 32);
    save(&mut hw, "long_wavy_back_light.png");
    drop(keep);
}

/// `[HR-02]` `[HAIR-04]` the eye macro (todo 1.7): lashes and brows at eye-macro distance in the
/// neutral studio (three-quarter and front) and in harsh side light. The lash pixels (records of
/// the lash groom with at least half a fibre of coverage) are measured against the skin and eye
/// around them: real lashes are much darker than the lid in a soft studio; they read as a pale
/// fan when an unshadowed light reaches them through the head.
#[test]
fn hr_06_eye_macro_lashes_are_dark_and_brows_dense() {
    use humanart_model::hair::{GroomKind, ScalpStyle};
    use humanart_render::camera::ReviewCamera;
    use humanart_render::hair::HairSettings;
    use humanart_render::hair::view::COVERAGE_ONE;
    use humanart_render::light::BuiltinRig;
    use humanart_render::renderer::RenderConfig;
    use template_head::{camera, harness, install, lum, rig, save};

    let set = template_head::grooms(ScalpStyle::Short, 100_000);
    let lash_groom = set.iter().position(|g| g.kind == GroomKind::Lashes).unwrap() as u32;
    let (w, h) = (900_u32, 600_u32);
    let aspect = w as f32 / h as f32;
    let mut hx = harness(RenderConfig::new(w, h));
    let keep = install(&mut hx, &set, HairSettings::default());
    let (_, hair) = template_head::fixture();
    // The left eye: the middle of its upper lid margin.
    let eye = hair.lines().upper_lids[0][hair.lines().upper_lids[0].len() / 2];
    for (name, kind, rig_kind, scale) in [
        ("eye_macro.png", ReviewCamera::ThreeQuarterLeft, BuiltinRig::NeutralStudio, 0.3),
        ("eye_macro_front.png", ReviewCamera::Front, BuiltinRig::NeutralStudio, 0.18),
        ("eye_macro_side_light.png", ReviewCamera::ThreeQuarterLeft, BuiltinRig::HarshSidelight, 0.3),
    ] {
        let r = rig(&mut hx, rig_kind);
        let mut cam = camera(kind, scale, aspect, &r);
        let offset = cam.position - cam.target;
        cam.target = eye;
        cam.position = eye + offset;
        hx.frames(&cam, &r, 32);
        save(&mut hx, name);
        let image = hx.resolved();
        let [_, coverage, records] = hx.renderer.hair().unwrap().visibility_buffers().unwrap();
        let coverage = read_f32(hx.gpu, coverage);
        let records = read_f32(hx.gpu, records);
        let lash: Vec<bool> = (0..image.len())
            .map(|i| {
                let cov = coverage[i].to_bits() as f32 / COVERAGE_ONE;
                records.get(4 * i + 1).is_some_and(|r| r.to_bits() >> 28 == lash_groom) && cov >= 0.5
            })
            .collect();
        let hair_px: Vec<bool> = coverage.iter().map(|c| c.to_bits() > 0).collect();
        // The box around the lash pixels, grown by half its size: the lids and the eye.
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for (i, _) in lash.iter().enumerate().filter(|(_, l)| **l) {
            let (x, y) = (i as u32 % w, i as u32 / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        let count = lash.iter().filter(|l| **l).count();
        assert!(count > 200, "{name}: {count} lash pixels");
        let (gx, gy) = ((x1 - x0) / 2, (y1 - y0) / 2);
        let (bx0, by0, bx1, by1) =
            (x0.saturating_sub(gx), y0.saturating_sub(gy), (x1 + gx).min(w - 1), (y1 + gy).min(h - 1));
        let mut around = Vec::new();
        for y in by0..=by1 {
            for x in bx0..=bx1 {
                let i = (y * w + x) as usize;
                if !hair_px[i] {
                    around.push(lum(image[i]));
                }
            }
        }
        around.sort_by(f32::total_cmp);
        let skin = around.get(around.len() / 2).copied().unwrap_or(0.0);
        let mut lashes: Vec<f32> = image.iter().zip(&lash).filter(|(_, l)| **l).map(|(p, _)| lum(*p)).collect();
        lashes.sort_by(f32::total_cmp);
        let median = lashes[lashes.len() / 2];
        let bright = lashes[lashes.len() * 9 / 10];
        println!(
            "[HAIR-04] {name}: {count} lash pixels, luminance median {median:.4} (90th percentile {bright:.4}) vs the \
             median of the skin and eye around them {skin:.4}: ratio {:.2} (90th {:.2})",
            median / skin.max(1e-6),
            bright / skin.max(1e-6)
        );
        if rig_kind == BuiltinRig::NeutralStudio {
            assert!(median < 0.5 * skin, "{name}: the lashes are not dark: {median} against {skin}");
        }
    }
    drop(keep);
}

/// `[HR-04]` GPU timings at 1440p of the medium straight groom set (100k scalp strands, brows,
/// lashes): the head close-up (three-quarter view filling the frame) and the far field (whole
/// body), each hair pass's scope and the whole frame (medians of 24 frames after 8 warm-up
/// frames), and the ribbon fallback on the close-up. One test so nothing else shares the GPU.
#[test]
fn hr_07_timings_at_1440p_close_up_and_far_field() {
    use humanart_model::hair::ScalpStyle;
    use humanart_render::camera::ReviewCamera;
    use humanart_render::hair::{HairPath, HairSettings};
    use humanart_render::light::BuiltinRig;
    use humanart_render::renderer::RenderConfig;
    use template_head::{camera, hair_ms, harness, install, rig, save};

    let set = template_head::grooms(ScalpStyle::MediumStraight, 100_000);
    let mut hz = harness(RenderConfig::new(2560, 1440));
    let keep = install(&mut hz, &set, HairSettings::default());
    let aspect = 2560.0 / 1440.0;
    let r = rig(&mut hz, BuiltinRig::NeutralStudio);
    let close = camera(ReviewCamera::ThreeQuarterLeft, 1.0, aspect, &r);
    let (t, _) = template_head::fixture();
    let (lo, hi) = t
        .mesh()
        .positions()
        .iter()
        .fold((humanart_math::Vec3::splat(f32::MAX), humanart_math::Vec3::splat(f32::MIN)), |(a, b), &p| {
            (a.min(p), b.max(p))
        });
    let far = humanart_render::camera::Camera {
        exposure: close.exposure,
        ..humanart_render::camera::Camera::review(ReviewCamera::FullBody, (lo + hi) * 0.5, 0.5 * (hi.y - lo.y), aspect)
    };
    for (name, cam) in [("close-up", close), ("far field", far)] {
        let (ms, scopes, frame_ms) = hair_ms(&mut hz, &cam, &r, 24);
        let lods: Vec<String> = (0..set.len())
            .filter_map(|i| hz.renderer.hair().and_then(|h| h.lod(i)))
            .map(|l| format!("keep {:.3} stride {}", l.keep, l.stride))
            .collect();
        println!(
            "[HR-04] 1440p {name}: hair passes {ms:.3} ms GPU median ({scopes}), whole frame {frame_ms:.3} ms; lod {lods:?}"
        );
        if name == "far field" {
            save(&mut hz, "far_field_1440p.png");
        }
    }
    // The close-up again without the hidden-segment test (before/after of the scene depth tiles).
    if let Some(hr) = hz.renderer.hair_mut() {
        hr.set_settings(&mut hz.resources, HairSettings { cull_hidden: false, ..HairSettings::default() }).unwrap();
    }
    let (ms, scopes, frame_ms) = hair_ms(&mut hz, &close, &r, 24);
    println!(
        "[HR-04] 1440p close-up, hidden segments not culled: hair passes {ms:.3} ms GPU median ({scopes}), whole frame \
         {frame_ms:.3} ms"
    );
    // The close-up again with one raster thread per segment (the walks not split).
    if let Some(hr) = hz.renderer.hair_mut() {
        hr.set_settings(&mut hz.resources, HairSettings { split_walks: false, ..HairSettings::default() }).unwrap();
    }
    let (ms, scopes, frame_ms) = hair_ms(&mut hz, &close, &r, 24);
    println!(
        "[HR-04] 1440p close-up, walks not split: hair passes {ms:.3} ms GPU median ({scopes}), whole frame {frame_ms:.3} ms"
    );
    // The ribbon fallback on the close-up.
    if let Some(hr) = hz.renderer.hair_mut() {
        hr.set_settings(&mut hz.resources, HairSettings { path: HairPath::Ribbon, ..HairSettings::default() }).unwrap();
    }
    let (ms, scopes, frame_ms) = hair_ms(&mut hz, &close, &r, 24);
    println!(
        "[HR-04] 1440p close-up, ribbon path: hair passes {ms:.3} ms GPU median ({scopes}), whole frame {frame_ms:.3} ms"
    );
    drop(keep);
}

/// `[HAIR-03]` the scalp presets on the template (todo 1.7), front, profile and back in the
/// neutral studio (the afro also from the front three-quarter, its silhouette): stills
/// `preset_<style>_<view>.png` in target/review/hair/. Each style's hair pixels are counted so a
/// style that draws nothing fails.
#[test]
fn hr_08_preset_stills_front_profile_back() {
    use humanart_model::hair::ScalpStyle;
    use humanart_render::camera::ReviewCamera;
    use humanart_render::hair::HairSettings;
    use humanart_render::light::BuiltinRig;
    use humanart_render::renderer::RenderConfig;
    use template_head::{camera, hair_mask, harness, install, rig, save};

    let (w, h) = (480_u32, 600_u32);
    let aspect = w as f32 / h as f32;
    let mut hx = harness(RenderConfig::new(w, h));
    for style in [
        ScalpStyle::Short,
        ScalpStyle::MediumStraight,
        ScalpStyle::LongWavy,
        ScalpStyle::Curly,
        ScalpStyle::Coily,
        ScalpStyle::Afro,
    ] {
        let set = template_head::grooms(style, 100_000);
        let keep = install(&mut hx, &set, HairSettings::default());
        let r = rig(&mut hx, BuiltinRig::NeutralStudio);
        // Long hair needs a wider frame.
        let scale = if style == ScalpStyle::LongWavy { 1.6 } else { 1.2 };
        for (view, kind) in [
            ("front", ReviewCamera::Front),
            ("three_quarter", ReviewCamera::ThreeQuarterLeft),
            ("profile", ReviewCamera::ProfileLeft),
            ("back", ReviewCamera::HairBack),
        ] {
            hx.frames(&camera(kind, scale, aspect, &r), &r, 24);
            save(&mut hx, &format!("preset_{}_{view}.png", style.name()));
            let covered = hair_mask(&hx).iter().filter(|m| **m).count();
            println!("[HAIR-03] preset {} {view}: {covered} hair pixels", style.name());
            assert!(covered > 2000, "{} {view}: {covered}", style.name());
        }
        if style == ScalpStyle::MediumStraight {
            // The front hairline and the part, close: a soft hairline, not a wig's edge.
            let (_, hair) = template_head::fixture();
            let brows: Vec<humanart_math::Vec3> = hair.lines().brows.iter().flatten().copied().collect();
            let mid = brows.iter().fold(humanart_math::Vec3::ZERO, |a, &b| a + b) * (1.0 / brows.len().max(1) as f32);
            let mut cam = camera(ReviewCamera::Front, 0.45, aspect, &r);
            let target = mid + hair.anchors().up * 0.05;
            let offset = cam.position - cam.target;
            cam.target = target;
            cam.position = target + offset;
            hx.frames(&cam, &r, 24);
            save(&mut hx, "hairline_medium-straight_front.png");
        }
        drop(keep);
    }
}
