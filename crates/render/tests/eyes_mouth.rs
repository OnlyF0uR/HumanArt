//! WP-16 eyes and mouth on the GPU: `[AC-WP16-2]` the analytic eye of `shaders/eye/optics.wgsl`
//! (f32, what the geometry pass traces) against its f64 reference
//! `humanart_model::eye::EyeGeometry`, ray by ray; `[AC-WP16-3]` what the eye and mouth shading
//! adds to the geometry pass at 1440p.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

mod common;

use humanart_gpu::resources::{BufferDesc, Category, Resources, Tag};
use humanart_math::{DVec3, Quat, Vec3};
use humanart_model::eye::{EyeFrame, EyeGeometry, EyeHit, EyeParams, LidAperture};
use humanart_model::mouth::{MouthOpening, MouthParams};
use humanart_render::camera::{Camera, Exposure, ReviewCamera};
use humanart_render::eye::{EYE_CLASS, EyeShading, FeatureTable, TEETH_CLASS, TONGUE_CLASS};
use humanart_render::light::{BuiltinRig, LightRig};
use humanart_render::mouth::MouthShading;
use humanart_render::renderer::RenderConfig;
use humanart_render::scene::{Material, Mesh, Scene};

fn f32_words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(gpu: &humanart_gpu::Gpu, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    common::wait(&gpu.device);
    let words = {
        let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    };
    staging.unmap();
    words
}

#[test]
fn ac_wp16_2_gpu_eye_trace_matches_the_f64_reference() {
    // Rays from a camera 60 mm in front of the eye at many angles and offsets, in the eye's frame
    // (millimetres, centre at the origin, axis +Z): the kind (sclera / cornea), the surface point
    // and the refracted iris-plane point must agree with the reference.
    let gpu = common::gpu();
    let params = EyeParams::default();
    let reference = EyeGeometry::new(&params);
    let mut cases = Vec::new();
    let mut rays = Vec::new();
    for i in 0..24 {
        for j in 0..24 {
            let origin = DVec3::new(-14.0 + 28.0 * f64::from(i) / 23.0, -14.0 + 28.0 * f64::from(j) / 23.0, 60.0);
            let target =
                DVec3::new(origin.x * 0.9 + f64::from(j % 5) - 2.0, origin.y * 0.9 + f64::from(i % 3) - 1.0, 0.0);
            let dir = (target - origin).normalize();
            rays.push((origin, dir));
            cases.extend([origin.x as f32, origin.y as f32, origin.z as f32, 0.0]);
            cases.extend([dir.x as f32, dir.y as f32, dir.z as f32, 0.0]);
        }
    }
    let count = rays.len();
    let shape = f32_words(&[
        0.0,
        0.0,
        0.0,
        params.eyeball_radius_mm,
        0.0,
        0.0,
        1.0,
        params.pupil_radius_mm(),
        0.0,
        1.0,
        0.0,
        params.limbus_radius_mm,
        params.cornea_radius_mm,
        params.cornea_back_radius_mm,
        params.cornea_thickness_mm,
        params.iris_depth_mm,
        humanart_model::eye::params::CORNEA_IOR,
        humanart_model::eye::params::AQUEOUS_IOR,
        0.0,
        0.0,
    ]);
    let mut resources = Resources::new(gpu, humanart_gpu::resources::Budget::UNLIMITED);
    let mut make = |bytes: &[u8], usage: wgpu::BufferUsages, what: &'static str| {
        let (id, _) = resources
            .create_buffer(&BufferDesc {
                size: bytes.len() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::shared(Category::Other, what),
            })
            .unwrap();
        resources.write_buffer(id, 0, bytes).unwrap();
        resources.buffer(id).unwrap().raw().clone()
    };
    let shape = make(&shape, wgpu::BufferUsages::UNIFORM, "eye probe shape");
    let input = make(&f32_words(&cases), wgpu::BufferUsages::STORAGE, "eye probe cases");
    let out = make(&vec![0; count * 32], wgpu::BufferUsages::STORAGE, "eye probe out");
    let source = humanart_shaders::compose("eye/probe.wgsl", &[]).unwrap().into_source();
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eye probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("eye probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eye probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: shape.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
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
    let (mut corneal, mut scleral, mut worst_point, mut worst_iris, mut mismatched) = (0, 0, 0.0_f64, 0.0_f64, 0);
    for (k, &(origin, dir)) in rays.iter().enumerate() {
        let g = &got[k * 8..k * 8 + 8];
        let kind = g[0].round() as u32;
        let point = DVec3::new(f64::from(g[4]), f64::from(g[5]), f64::from(g[6]));
        let iris = DVec3::new(f64::from(g[1]), f64::from(g[2]), f64::from(g[3]));
        match reference.trace(origin, dir) {
            EyeHit::Miss => mismatched += usize::from(kind != 0),
            EyeHit::Sclera { point: p, .. } => {
                if kind == 1 {
                    scleral += 1;
                    worst_point = worst_point.max(p.distance(point));
                } else {
                    mismatched += 1;
                }
            }
            EyeHit::Cornea { point: p, iris: i, .. } => {
                if kind == 2 {
                    corneal += 1;
                    worst_point = worst_point.max(p.distance(point));
                    worst_iris = worst_iris.max(i.distance(iris));
                } else {
                    mismatched += 1;
                }
            }
        }
    }
    println!(
        "[AC-WP16-2] GPU eye trace vs f64: {count} rays, {corneal} corneal, {scleral} scleral, {mismatched} \
         classified differently; worst surface point {worst_point:.2e} mm, worst iris point {worst_iris:.2e} mm"
    );
    assert!(corneal > 50 && scleral > 50, "the rays cover both surfaces");
    // Rays grazing the limbus may fall either side in f32.
    assert!(mismatched <= 4, "{mismatched} rays classified differently");
    assert!(worst_point < 1e-3, "surface points agree to 1 um ({worst_point} mm)");
    assert!(worst_iris < 5e-3, "iris points agree to 5 um ({worst_iris} mm)");
}

/// Median of `values` (sorted in place).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// The median GPU milliseconds of the geometry pass over `frames` frames after a warm-up.
fn geometry_ms(harness: &mut common::Harness, camera: &Camera, rig: &LightRig, frames: u32) -> f64 {
    let mut samples = Vec::new();
    for i in 0..frames + 20 {
        harness.frame(camera, rig, &[], false, None, i == 0);
        common::wait(&harness.gpu.device);
        if harness.renderer.collect_timings() && i >= 20 {
            let report = harness.renderer.timings();
            assert!(report.gpu_timed);
            if let Some(scope) = report.scopes.iter().find(|s| s.name == "geometry") {
                samples.push(scope.gpu_nanos.unwrap_or(0) as f64 / 1e6);
            }
        }
    }
    assert!(samples.len() as u32 >= frames / 2, "only {} geometry timings arrived", samples.len());
    median(&mut samples)
}

#[test]
fn ac_wp16_3_eye_and_mouth_shading_cost_at_1440p() {
    // [AC-WP16-3] The analytic eye (traced sclera / cornea / refracted iris, catchlights, lid
    // film) and the mouth materials cost little on top of the geometry pass at 2560x1440, Review
    // quality, three-light studio rig with shadows: a head-sized ellipsoid with two eyeballs,
    // teeth and a tongue, timed with the feature table filled and with it empty (the same
    // objects then shade as plain materials). Portrait framing must stay within 0.5 ms; the eye
    // macro (an eyeball filling the frame, the worst case) is printed for the record.
    let head = Vec3::new(0.0, 1.6, 0.0);
    let mut scene = Scene::default();
    scene.add(Mesh::ellipsoid(head, Vec3::new(0.075, 0.11, 0.095), 96, 64), Material::default());
    let eye_material = Material { class: EYE_CLASS, ..Material::default() };
    let eyes = [Vec3::new(0.032, 1.63, 0.075), Vec3::new(-0.032, 1.63, 0.075)];
    let eye_objects = eyes.map(|c| scene.add(Mesh::sphere(c, 0.012, 64, 32), eye_material));
    let teeth = scene.add(
        Mesh::ellipsoid(Vec3::new(0.0, 1.53, 0.078), Vec3::new(0.024, 0.006, 0.012), 48, 24),
        Material { class: TEETH_CLASS, ..Material::default() },
    );
    let tongue = scene.add(
        Mesh::ellipsoid(Vec3::new(0.0, 1.522, 0.07), Vec3::new(0.02, 0.005, 0.02), 48, 24),
        Material { class: TONGUE_CLASS, ..Material::default() },
    );
    let mut harness = common::Harness::new(RenderConfig::new(2560, 1440), &scene);
    assert!(harness.gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY), "timestamp queries");
    let eye = |centre: Vec3, partner: Option<usize>| EyeShading {
        frame: EyeFrame { centre, radius: 0.012, axis: Vec3::Z },
        head: Quat::IDENTITY,
        params: EyeParams::default(),
        aperture: LidAperture::default(),
        lashes: 1.0,
        partner,
    };
    let mut table = FeatureTable::new();
    let left = table.add(&eye(eyes[0], None).entry().unwrap()).unwrap();
    let right = table.add(&eye(eyes[1], Some(left)).entry().unwrap()).unwrap();
    table.set(left, &eye(eyes[0], Some(right)).entry().unwrap()).unwrap();
    let mouth = MouthShading {
        opening: MouthOpening {
            centre: Vec3::new(0.0, 1.528, 0.088),
            forward: Vec3::Z,
            across: Vec3::X,
            half_width: 0.025,
            half_height: 0.008,
        },
        up: Vec3::Y,
        upper_edge: Vec3::new(0.0, 1.528, 0.09),
        lower_edge: Vec3::new(0.0, 1.527, 0.09),
        crown_height: 0.01,
        params: MouthParams::default(),
    };
    let mouth_entry = table.add(&mouth.entry().unwrap()).unwrap();
    table.assign(eye_objects[0], Some(left)).unwrap();
    table.assign(eye_objects[1], Some(right)).unwrap();
    table.assign(teeth, Some(mouth_entry)).unwrap();
    table.assign(tongue, Some(mouth_entry)).unwrap();
    let empty = FeatureTable::new();
    let rig = LightRig::builtin(BuiltinRig::NeutralStudio, head, 0.12);
    let aspect = 2560.0 / 1440.0;
    let framings = [
        ("portrait", Camera::review(ReviewCamera::Front, head, 0.13, aspect)),
        ("eye macro", Camera::review(ReviewCamera::Front, eyes[0] + Vec3::Z * 0.007, 0.012, aspect)),
    ];
    for (name, mut camera) in framings {
        camera.exposure = Exposure::ev100(rig.ev100);
        let resources = &mut harness.resources;
        harness.renderer.set_features(resources, &empty).unwrap();
        let plain = geometry_ms(&mut harness, &camera, &rig, 90);
        let resources = &mut harness.resources;
        harness.renderer.set_features(resources, &table).unwrap();
        let shaded = geometry_ms(&mut harness, &camera, &rig, 90);
        let extra = shaded - plain;
        println!(
            "[AC-WP16-3] 2560x1440 {name}: geometry pass {plain:.3} ms plain, {shaded:.3} ms with eyes and mouth ({extra:+.3} ms)"
        );
        if name == "portrait" {
            assert!(extra <= 0.5, "{name}: the eyes and mouth add {extra:.3} ms to the geometry pass");
        }
    }
}
