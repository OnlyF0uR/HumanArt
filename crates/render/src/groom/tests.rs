//! GPU tests of the hair generation on this machine's GPU (headless): parity of every modifier
//! with the CPU reference `humanart_model::hair::generate` and the groom edit latency for 100 000
//! strands `[AC-WP17-2]`, attach to a moved skin, and a debug picture of the GPU groom.

use std::sync::OnceLock;
use std::time::Instant;

use humanart_core::rng::Rng;
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_math::{F16, Quat, Vec2, Vec3, Vec4};
use humanart_model::hair::generate::{GenerateInput, attach, generate, root_frames};
use humanart_model::hair::{
    GroomLayout, GroomMap, GroomMaps, GroomParams, Guides, ProxyHead, ScalpMask, SdfGrid, TriangleGrid, place,
};

use super::{GpuGroom, GroomCapacity};

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU"))
}

fn read_words(buffer: &wgpu::Buffer) -> Vec<u32> {
    let gpu = gpu();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("groom readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("the readback maps"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("the device answers");
    let words = {
        let bytes = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
    };
    staging.unmap();
    words
}

/// Guides that grow out along the normal and fall under gravity, `count` of them on the scalp.
fn guides_on(head: &ProxyHead, mask: &ScalpMask, count: usize, length: f32, seed: u64) -> Guides {
    let mut g = Guides::new(12).unwrap();
    let mut rng = Rng::new(seed);
    let tris = mask.triangles();
    while g.len() < count {
        let t = tris[rng.index(tris.len())];
        let r1 = rng.f32().sqrt();
        let r2 = rng.f32();
        let bary = [r1 * (1.0 - r2), r1 * r2];
        let f = head.surface.frame(t, bary).unwrap();
        let mut world = vec![f.origin];
        let mut dir = f.n;
        let mut p = f.origin;
        let len = length * rng.range_f32(0.8, 1.2);
        for _ in 0..11 {
            dir = (dir + Vec3::new(0.0, -0.45, -0.1)).normalize();
            p += dir * (len / 11.0);
            world.push(p);
        }
        let local: Vec<Vec3> = world.iter().map(|&w| f.to_local(w)).collect();
        g.push(t, bary, &local).unwrap();
    }
    g
}

/// A head, its scalp, guides, the distance grid, a layout and a parameter set with every modifier on.
struct Scene {
    head: ProxyHead,
    mask: ScalpMask,
    guides: Guides,
    sdf: SdfGrid,
    params: GroomParams,
    layout: GroomLayout,
}

fn scene(strands: u32, guides: usize, rings: u32) -> Scene {
    let head = ProxyHead::new(rings).unwrap();
    let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
    let g = guides_on(&head, &mask, guides, 0.16, 21);
    let grid = TriangleGrid::new(head.surface.positions(), head.surface.triangles(), 0.01).unwrap();
    let sdf = SdfGrid::build(&grid, 0.004, 0.03).unwrap();
    let top = head.surface_point(Vec3::new(0.0, 1.0, -0.3));
    let params = GroomParams {
        strand_count: strands as f32,
        curl_radius: 0.003,
        curl_frequency: 40.0,
        wave_amplitude: 0.002,
        frizz: 0.001,
        flyaway_fraction: 0.05,
        noise: 0.1,
        parting_x: 0.01,
        parting_y: 1.6,
        parting_strength: 0.8,
        parting_width: 0.025,
        cowlick0_x: top.x,
        cowlick0_y: top.y,
        cowlick0_z: top.z,
        cowlick0_radius: 0.04,
        cowlick0_strength: -0.8,
        grey_fraction: 0.3,
        clump_count: 300.0,
        ..GroomParams::default()
    };
    let mut direction = GroomMap::new(8, 8, 2, &[0.0, 0.0]).unwrap();
    let mut length = GroomMap::new(8, 8, 1, &[1.0]).unwrap();
    for y in 0..8 {
        for x in 0..8 {
            direction.texel_mut(x, y).unwrap().copy_from_slice(&[0.5 * (x as f32 / 7.0), 0.3]);
            length.texel_mut(x, y).unwrap()[0] = 0.6 + 0.1 * y as f32;
        }
    }
    let maps = GroomMaps { direction: Some(direction), length: Some(length), ..GroomMaps::default() };
    let layout = place(&head.surface, &mask, &g, &params, &maps, 99).unwrap();
    Scene { head, mask, guides: g, sdf, params, layout }
}

fn build(s: &Scene, res: &mut Resources) -> GpuGroom {
    let capacity = GroomCapacity { strands: s.layout.len() as u32, guides: s.guides.len() as u32, points: 16 };
    let mut groom = GpuGroom::new(gpu(), res, &s.head.surface, Some(&s.sdf), capacity).unwrap();
    groom.set_params(res, &s.params).unwrap();
    groom.set_guides(res, &s.guides.resampled(16)).unwrap();
    groom.set_layout(res, &s.head.surface, &s.layout).unwrap();
    groom
}

fn submit(f: impl FnOnce(&mut wgpu::CommandEncoder) -> bool) {
    let gpu = gpu();
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("groom test") });
    assert!(f(&mut encoder));
    gpu.queue.submit([encoder.finish()]);
}

fn vec4s(words: &[u32], n: usize) -> Vec<Vec4> {
    words.as_chunks::<4>().0.iter().take(n).map(|c| Vec4::from_array(c.map(f32::from_bits))).collect()
}

/// The tolerance of one stored f16 component: two units in the last place plus the subnormal
/// range (the GPU may flush f16 subnormals, below 6.1e-5).
fn f16_tolerance(v: f32) -> f32 {
    v.abs() * (2.0 / 1024.0) + 6.2e-5
}

// [AC-WP17-2]: every stage of the GPU generation against the CPU reference, all modifiers on.
#[test]
fn ac_wp17_2_gpu_modifiers_match_the_cpu_reference() {
    let s = scene(20_000, 300, 48);
    let mut res = Resources::new(gpu(), Budget::UNLIMITED);
    let mut groom = build(&s, &mut res);
    submit(|e| groom.encode_generate(e));
    let curves = s.guides.resampled(16);
    let frames = root_frames(&s.layout, &s.head.surface).unwrap();
    let input =
        GenerateInput { layout: &s.layout, curves: &curves, params: &s.params, frames: &frames, sdf: Some(&s.sdf) };
    let cpu = generate(&input).unwrap();
    let n = s.layout.len() * 16;
    assert_eq!(cpu.strands.len(), 20_000);

    // Stage 1: world points.
    let shape = vec4s(&read_words(groom.outputs().shape), n);
    let shape_err = shape.iter().zip(&cpu.shape).map(|(g, c)| (g.truncate() - *c).length()).fold(0.0_f32, f32::max);
    // Stage 2: stored f16 offsets.
    let stored = read_words(groom.outputs().stored);
    let mut stored_bad = 0;
    let mut stored_err = 0.0_f32;
    for p in 0..n {
        let (a, b) = (stored[2 * p], stored[2 * p + 1]);
        let gpu_v = [
            F16::from_bits(a as u16).to_f32(),
            F16::from_bits((a >> 16) as u16).to_f32(),
            F16::from_bits(b as u16).to_f32(),
        ];
        let cpu_v = [cpu.strands.offsets[3 * p], cpu.strands.offsets[3 * p + 1], cpu.strands.offsets[3 * p + 2]];
        for (g, c) in gpu_v.iter().zip(cpu_v) {
            let c = c.to_f32();
            stored_err = stored_err.max((g - c).abs());
            if (g - c).abs() > f16_tolerance(c) {
                stored_bad += 1;
            }
        }
    }
    // Stage 2: attributes.
    let attrs = read_words(groom.outputs().attrs);
    let mut attr_err = 0.0_f32;
    for i in 0..s.layout.len() {
        let w = &attrs[8 * i..8 * i + 8];
        let st = &cpu.strands;
        attr_err = attr_err.max((f32::from_bits(w[0]) - st.length[i]).abs() / st.length[i].max(1e-6));
        assert!((f32::from_bits(w[1]) - st.width_root[i]).abs() < 1e-9, "width root {i}");
        assert!((f32::from_bits(w[2]) - st.width_tip[i]).abs() < 1e-9, "width tip {i}");
        assert_eq!(f32::from_bits(w[3]), st.curl_phase[i], "curl phase {i}");
        assert_eq!(w[4] & 0xFFFF, u32::from(st.color_id[i]), "colour id {i}");
        assert_eq!(w[4] >> 16, u32::from(st.grey[i]), "grey {i}");
        assert!((f32::from_bits(w[5]) - st.stiffness[i]).abs() < 1e-6, "stiffness {i}");
        assert_eq!(w[6], st.clump[i], "clump {i}");
    }
    // Stage 3 on the rest skin.
    groom.bind_rest_skin(&gpu().device).unwrap();
    submit(|e| groom.encode_attach(e));
    let world = vec4s(&read_words(groom.outputs().world), n);
    let mut cpu_world = vec![Vec4::ZERO; n];
    attach(&cpu.strands, &s.head.surface, s.head.surface.positions(), s.head.surface.normals(), &mut cpu_world)
        .unwrap();
    let attach_err =
        world.iter().zip(&cpu_world).map(|(g, c)| (g.truncate() - c.truncate()).length()).fold(0.0_f32, f32::max);
    let width_err = world.iter().zip(&cpu_world).map(|(g, c)| (g.w - c.w).abs()).fold(0.0_f32, f32::max);
    println!(
        "[AC-WP17-2] parity, {} strands x 16 points on {}: shape max {shape_err:.3e} m, stored max {stored_err:.3e} \
         ({stored_bad} components beyond 2 f16 ulp), length rel {attr_err:.3e}, attach max {attach_err:.3e} m",
        s.layout.len(),
        gpu().adapter_info().name
    );
    assert!(shape_err < 2e-5, "shape {shape_err}");
    assert_eq!(stored_bad, 0);
    assert!(attr_err < 1e-4, "length {attr_err}");
    assert!(attach_err < 1e-4 && width_err < 1e-9, "attach {attach_err} {width_err}");
    assert_eq!(groom.late_pipelines(), 0);
    groom.destroy(&mut res);
}

// The strands ride a moved and rotated skin: attach on the GPU against the CPU attach.
#[test]
fn attach_follows_a_deformed_skin() {
    let s = scene(3000, 100, 32);
    let mut res = Resources::new(gpu(), Budget::UNLIMITED);
    let mut groom = build(&s, &mut res);
    submit(|e| groom.encode_generate(e));
    let curves = s.guides.resampled(16);
    let frames = root_frames(&s.layout, &s.head.surface).unwrap();
    let input =
        GenerateInput { layout: &s.layout, curves: &curves, params: &s.params, frames: &frames, sdf: Some(&s.sdf) };
    let cpu = generate(&input).unwrap();
    // Rotate the head 30 degrees about Y, nod it and stretch the top: a deformation.
    let q = Quat::from_axis_angle(Vec3::Y, 0.5) * Quat::from_axis_angle(Vec3::X, 0.2);
    let c = s.head.centre;
    let positions: Vec<Vec3> = s
        .head
        .surface
        .positions()
        .iter()
        .map(|&p| {
            let d = p - c;
            c + q * (d + Vec3::new(0.0, d.y.max(0.0) * 0.1, 0.0)) + Vec3::new(0.05, 0.0, 0.0)
        })
        .collect();
    let normals: Vec<Vec3> = s.head.surface.normals().iter().map(|&n| q * n).collect();
    let bytes = |v: &[Vec3]| -> Vec<u8> { v.iter().flat_map(|p| [p.x, p.y, p.z]).flat_map(f32::to_le_bytes).collect() };
    let make = |data: &[u8]| {
        let b = gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deformed skin"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu().queue.write_buffer(&b, 0, data);
        b
    };
    let (pb, nb) = (make(&bytes(&positions)), make(&bytes(&normals)));
    groom.bind_skin(&gpu().device, &pb, &nb).unwrap();
    submit(|e| groom.encode_attach(e));
    let n = s.layout.len() * 16;
    let world = vec4s(&read_words(groom.outputs().world), n);
    let mut cpu_world = vec![Vec4::ZERO; n];
    attach(&cpu.strands, &s.head.surface, &positions, &normals, &mut cpu_world).unwrap();
    let err = world.iter().zip(&cpu_world).map(|(g, c)| (g.truncate() - c.truncate()).length()).fold(0.0_f32, f32::max);
    assert!(err < 1e-4, "{err}");
    // Roots sit on the moved skin.
    for i in 0..s.layout.len() {
        let root = s.head.surface.frame_on(s.layout.triangle[i], s.layout.bary[i], &positions, &normals).unwrap();
        assert!((world[i * 16].truncate() - root.origin).length() < 1e-5);
    }
    assert!(groom.bind_skin(&gpu().device, &make(&[0; 16]), &nb).is_err());
    groom.destroy(&mut res);
}

/// Milliseconds between two resolved timestamps.
fn timed_ms(record: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::QuerySet)) -> f64 {
    let gpu = gpu();
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    record(&mut encoder, &queries);
    encoder.resolve_query_set(&queries, 0..2, &resolve, 0);
    gpu.queue.submit([encoder.finish()]);
    let w = read_words(&resolve);
    let t0 = u64::from(w[0]) | (u64::from(w[1]) << 32);
    let t1 = u64::from(w[2]) | (u64::from(w[3]) << 32);
    t1.saturating_sub(t0) as f64 * f64::from(gpu.queue.get_timestamp_period()) * 1e-6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

// [AC-WP17-2]: groom edit latency for 100 000 strands. The specification has no row for a groom
// edit in section 14.2; the binding budgets are "interactive frame rate during any edit >= 60 fps"
// (16.7 ms per frame for everything) and the hair render budget of 2.5 ms. The regeneration is
// asserted to fit one 60 fps frame; the numbers are printed for the status file.
#[test]
fn ac_wp17_2_groom_edit_latency_100k_strands() {
    let gpu = gpu();
    assert!(gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY), "no timestamp queries: {}", gpu.report());
    let mut s = scene(100_000, 1000, 96);
    assert_eq!(s.layout.len(), 100_000);
    let mut res = Resources::new(gpu, Budget::UNLIMITED);
    let mut groom = build(&s, &mut res);
    groom.bind_rest_skin(&gpu.device).unwrap();
    let (mut generate_ms, mut attach_ms, mut cpu_ms) = (vec![], vec![], vec![]);
    for round in 0..20 {
        // A guide edit: turn 50 guides a little (a comb stroke), then re-upload and regenerate.
        let start = Instant::now();
        for g in 0..50 {
            let i = (round * 50 + g) % s.guides.len();
            let rot = Quat::from_axis_angle(Vec3::Z, 0.05);
            for p in s.guides.offsets_mut(i) {
                *p = rot * *p;
            }
        }
        let curves = s.guides.resampled(16);
        groom.set_guides(&mut res, &curves).unwrap();
        let cpu = start.elapsed().as_secs_f64() * 1e3;
        generate_ms.push(timed_ms(|e, q| assert!(groom.encode_generate_timed(e, Some((q, 0))))));
        cpu_ms.push(cpu);
        attach_ms.push(timed_ms(|e, q| assert!(groom.encode_attach_timed(e, Some((q, 0))))));
    }
    // A density / count edit: placement on the CPU plus the layout upload.
    let start = Instant::now();
    let layout = place(&s.head.surface, &s.mask, &s.guides, &s.params, &GroomMaps::default(), 5).unwrap();
    let place_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    groom.set_layout(&mut res, &s.head.surface, &layout).unwrap();
    let upload_ms = start.elapsed().as_secs_f64() * 1e3;
    s.layout = layout;
    let (g, a, c) = (median(generate_ms), median(attach_ms), median(cpu_ms));
    println!(
        "[AC-WP17-2] 100k strands x 16 points, 1000 guides, {} ({:?}): regenerate {g:.3} ms GPU, attach {a:.3} ms GPU, \
         guide edit CPU (50 guides + resample 1000 + upload) {c:.3} ms; placement {place_ms:.1} ms CPU + layout \
         upload {upload_ms:.1} ms (debug or release per the build profile)",
        gpu.adapter_info().name,
        gpu.adapter_info().backend
    );
    assert!(g + c < 16.7, "a guide edit must regenerate within one 60 fps frame: {g} + {c} ms");
    groom.destroy(&mut res);
}

// Debug picture: the GPU groom seen from the front-left, drawn by a small strand rasteriser, written
// to target/agents/hair/groom_gpu.png for a human (and the agent) to look at.
#[test]
fn groom_debug_picture() {
    let s = scene(30_000, 400, 64);
    let mut res = Resources::new(gpu(), Budget::UNLIMITED);
    let mut groom = build(&s, &mut res);
    groom.bind_rest_skin(&gpu().device).unwrap();
    submit(|e| groom.encode_generate(e) && groom.encode_attach(e));
    let n = s.layout.len() * 16;
    let world = vec4s(&read_words(groom.outputs().world), n);
    let (w, h) = (640_usize, 480_usize);
    let mut rgb = vec![30_u8; w * h * 3];
    let view = Quat::from_axis_angle(Vec3::Y, -0.7);
    let project = |p: Vec3| -> Vec2 {
        let v = view * (p - s.head.centre);
        Vec2::new(w as f32 * 0.25 + v.x * 1500.0 + 160.0, h as f32 * 0.5 - v.y * 1500.0)
    };
    // Head silhouette: the surface vertices.
    for &p in s.head.surface.positions() {
        let q = project(p);
        let (x, y) = (q.x as i64, q.y as i64);
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 3;
            rgb[i..i + 3].copy_from_slice(&[200, 160, 140]);
        }
    }
    for strand in world.chunks(16) {
        for k in 1..16 {
            let (a, b) = (project(strand[k - 1].truncate()), project(strand[k].truncate()));
            let steps = (b - a).length().ceil().max(1.0) as usize;
            for t in 0..=steps {
                let p = a + (b - a) * (t as f32 / steps as f32);
                let (x, y) = (p.x as i64, p.y as i64);
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    let i = (y as usize * w + x as usize) * 3;
                    let shade = 60 + (k * 8) as u8;
                    rgb[i] = rgb[i].saturating_add(shade / 6);
                    rgb[i + 1] = rgb[i + 1].saturating_add(shade / 9);
                    rgb[i + 2] = rgb[i + 2].saturating_add(shade / 14);
                }
            }
        }
    }
    let image = humanart_io::Image::new(
        humanart_io::Format::Png,
        w as u32,
        h as u32,
        3,
        humanart_io::ColorSpace::Srgb,
        humanart_io::Samples::U8(rgb),
    )
    .unwrap();
    let png = humanart_io::png::encode(&image).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/hair");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("groom_gpu.png"), png).unwrap();
    groom.destroy(&mut res);
}
