//! GPU tests of the skin compositor, the micro-detail generator and the streaming integration, on
//! the real adapter, headless. CPU references: `humanart_model::skin_appearance`.

use humanart_core::rng::Rng;
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::streaming::TileRequest;
use humanart_gpu::{Capability, Gpu, GpuError};
use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{f16_bits_to_f32, f32_to_f16_bits};
use humanart_model::skin_appearance::antitile::GaussianExemplar;
use humanart_model::skin_appearance::channels::{ATTRIBUTE_COUNT, Attribute, BaseValues};
use humanart_model::skin_appearance::dirty::Rect;
use humanart_model::skin_appearance::features::{GeneratorKind, GeneratorParams, Target, generate};
use humanart_model::skin_appearance::layers::{
    BlendMode, Layer, LayerContent, LayerId, LayerKind, LayerStack, TexelInputs, composite_texel,
};
use humanart_model::skin_appearance::microdetail::{ExemplarPlan, MicroDetailParams};
use humanart_model::skin_appearance::pigment::{Mapping, PigmentModel};
use humanart_model::skin_appearance::resolve::resolve_texel;
use humanart_model::skin_appearance::zones::{ZONE_SLOTS, Zone, ZoneTable};

use super::microdetail::MicroDetailGpu;
use humanart_model::eval::WrinkleDriver;
use humanart_model::skin_appearance::wrinkle::{DriverTable, WrinkleMap, WrinkleMaps, blend_texel};

use super::stream::{SkinStreaming, TileSource, TileView};
use super::{SkinCompositor, SkinDesc};

fn gpu() -> &'static Gpu {
    // The process's shared device (ADR-040).
    humanart_gpu::shared_headless().unwrap_or_else(|e| panic!("this machine has a GPU: {e}"))
}

/// Reads one array layer of a `COPY_SRC` texture, rows tightly packed.
fn read_layer(gpu: &Gpu, texture: &wgpu::Texture, layer: u32) -> Vec<u8> {
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    let (w, h) = (texture.width(), texture.height());
    let row = w * texel;
    let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skin readback"),
        size: u64::from(padded) * u64::from(h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap_or_else(|e| panic!("{e}"));
    rx.recv().unwrap_or_else(|e| panic!("{e}")).unwrap_or_else(|e| panic!("{e}"));
    let mapped = staging.slice(..).get_mapped_range().unwrap_or_else(|e| panic!("{e}")).to_vec();
    staging.unmap();
    let mut out = Vec::with_capacity((row * h) as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&mapped[y * padded as usize..y * padded as usize + row as usize]);
    }
    out
}

fn halves(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|b| f16_bits_to_f32(u16::from_le_bytes(*b))).collect()
}

fn submit(gpu: &Gpu, encoder: wgpu::CommandEncoder) {
    gpu.queue.submit([encoder.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap_or_else(|e| panic!("{e}"));
}

/// A value exactly representable in f16 (multiples of 1/256 in 0..2).
fn q(v: f32) -> f32 {
    (v * 256.0).round() / 256.0
}

/// The test scene: two 64² tiles, five layers of every kind, zone masks and baked inputs.
struct Scene {
    stack: LayerStack,
    base: BaseValues,
    baked: Vec<Vec<u8>>,
    zones: Vec<[Vec<u8>; 3]>,
}

const SIZE: u32 = 64;

fn scene() -> Scene {
    let mut stack = LayerStack::new(2, SIZE).unwrap_or_else(|e| panic!("{e}"));
    stack
        .push(Layer::fill(
            LayerId(1),
            "tan",
            BlendMode::Add,
            &[(Attribute::Melanin, 0.15), (Attribute::Haemoglobin, 0.05)],
        ))
        .unwrap_or_else(|e| panic!("{e}"));
    let mut moles = Layer::content(
        LayerId(2),
        "moles",
        BlendMode::Mix,
        [Some(Attribute::Melanin), Some(Attribute::EumelaninRatio), Some(Attribute::Height), None],
    );
    let mut c = LayerContent::new(0, Rect::new(8, 8, 40, 30)).unwrap_or_else(|e| panic!("{e}"));
    let mut rng = Rng::new(3);
    for (v, cov) in c.values.iter_mut().zip(c.coverage.iter_mut()) {
        *v = [q(rng.f32()), q(rng.f32()), q(rng.range_f32(-0.5, 0.5)), 0.0];
        *cov = q(rng.f32());
    }
    if let LayerKind::Content { tiles, .. } = &mut moles.kind {
        tiles.push(c);
    }
    stack.push(moles).unwrap_or_else(|e| panic!("{e}"));
    let mut lips = Layer::fill(
        LayerId(3),
        "lip colour",
        BlendMode::Mix,
        &[(Attribute::OverrideRed, 0.6), (Attribute::OverrideGreen, 0.1), (Attribute::OverrideCoverage, 1.0)],
    );
    lips.opacity = 0.6;
    lips.zone = Some(Zone::Lips);
    stack.push(lips).unwrap_or_else(|e| panic!("{e}"));
    let mut wet = Layer::content(
        LayerId(4),
        "sweat",
        BlendMode::Add,
        [
            Some(Attribute::Haemoglobin),
            Some(Attribute::Wetness),
            Some(Attribute::DetailNormalX),
            Some(Attribute::Cavity),
        ],
    );
    let mut c = LayerContent::new(1, Rect::new(0, 0, SIZE, SIZE)).unwrap_or_else(|e| panic!("{e}"));
    for (v, cov) in c.values.iter_mut().zip(c.coverage.iter_mut()) {
        *v = [q(rng.range_f32(-0.2, 0.3)), q(rng.f32()), q(rng.range_f32(-1.0, 1.0)), q(rng.range_f32(-0.5, 0.0))];
        *cov = q(rng.f32());
    }
    if let LayerKind::Content { tiles, .. } = &mut wet.kind {
        tiles.push(c);
    }
    stack.push(wet).unwrap_or_else(|e| panic!("{e}"));
    let mut hidden = Layer::fill(LayerId(5), "hidden", BlendMode::Max, &[(Attribute::Melanin, 1.0)]);
    hidden.enabled = false;
    stack.push(hidden).unwrap_or_else(|e| panic!("{e}"));

    let mut base = BaseValues::default();
    base.set_target("skin.melanin", 0.35);
    base.set_target("skin.wetness", 0.2);
    let n = (SIZE * SIZE) as usize;
    let mut baked = Vec::new();
    let mut zones = Vec::new();
    for tile in 0..2 {
        let mut b = vec![0_u8; n * 4];
        let mut z = [vec![0_u8; n * 4], vec![0_u8; n * 4], vec![0_u8; n * 4]];
        for i in 0..n {
            let (x, y) = ((i as u32 % SIZE) as u8, (i as u32 / SIZE) as u8);
            b[i * 4] = x * 4;
            b[i * 4 + 1] = 255 - y * 2;
            if tile == 0 {
                z[0][i * 4] = x * 4; // cheeks
                z[0][i * 4 + 3] = y * 4; // lips
                z[2][i * 4] = (x ^ y).wrapping_mul(9); // temples
            } else {
                z[1][i * 4 + 1] = 200; // elbows
            }
        }
        baked.push(b);
        zones.push(z);
    }
    Scene { stack, base, baked, zones }
}

fn compositor(scene: &Scene) -> SkinCompositor {
    let gpu = gpu();
    let mut compositor = SkinCompositor::new(gpu, SkinDesc { tiles: 2, tile_size: SIZE, atlas_size: 256 })
        .unwrap_or_else(|e| panic!("{e}"));
    compositor.set_globals(gpu, &scene.base, &ZoneTable::default(), &Mapping::DEFAULT);
    for tile in 0..2 {
        let z = &scene.zones[tile];
        compositor
            .upload_inputs(gpu, tile as u32, &scene.baked[tile], [&z[0], &z[1], &z[2]])
            .unwrap_or_else(|e| panic!("{e}"));
    }
    compositor.sync_layers(gpu, &scene.stack).unwrap_or_else(|e| panic!("{e}"));
    compositor
}

fn texel_inputs(scene: &Scene, tile: usize, i: usize) -> TexelInputs {
    let mut zones = [0.0; ZONE_SLOTS];
    for (k, layer) in scene.zones[tile].iter().enumerate() {
        for c in 0..4 {
            zones[k * 4 + c] = f32::from(layer[i * 4 + c]) / 255.0;
        }
    }
    TexelInputs {
        zones,
        thickness: f32::from(scene.baked[tile][i * 4]) / 255.0,
        cavity: f32::from(scene.baked[tile][i * 4 + 1]) / 255.0,
        variation: f32::from(scene.baked[tile][i * 4 + 2]) / 255.0,
    }
}

/// The planes of a tile as attributes per texel.
fn read_planes(compositor: &SkinCompositor, tile: u32) -> Vec<[f32; ATTRIBUTE_COUNT]> {
    let gpu = gpu();
    let n = (SIZE * SIZE) as usize;
    let mut out = vec![[0.0; ATTRIBUTE_COUNT]; n];
    for plane in 0..5 {
        let values = halves(&read_layer(gpu, compositor.planes(), tile * 5 + plane));
        for (i, texel) in out.iter_mut().enumerate() {
            for c in 0..4 {
                texel[plane as usize * 4 + c] = values[i * 4 + c];
            }
        }
    }
    out
}

#[test]
fn skn02_gpu_compositor_matches_the_cpu_reference() {
    // [SKN-02] [SKN-10] [SKN-11] [SKN-12] [SKN-31] [SKN-32]: both passes on the device against the
    // CPU reference, every texel of both tiles, every attribute and resolved channel.
    let gpu = gpu();
    let mut scene = scene();
    let mut compositor = compositor(&scene);
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let report = compositor.resolve(gpu, &mut encoder, &mut scene.stack, None).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    assert_eq!(report.texels, 2 * u64::from(SIZE * SIZE), "everything starts dirty");
    assert_eq!(compositor.late_pipelines(), 0);
    let model = PigmentModel::default();
    let zones = ZoneTable::default();
    let mut worst = [0.0_f32; 5];
    for tile in 0..2_u32 {
        let planes = read_planes(&compositor, tile);
        let resolved: Vec<Vec<f32>> =
            (0..4).map(|k| halves(&read_layer(gpu, compositor.resolved(), tile * 4 + k))).collect();
        let specular = read_layer(gpu, compositor.specular(), tile);
        for (i, gpu_v) in planes.iter().enumerate() {
            let inputs = texel_inputs(&scene, tile as usize, i);
            let (x, y) = (i as u32 % SIZE, i as u32 / SIZE);
            let cpu_v = composite_texel(&scene.stack, &scene.base, &zones, &inputs, tile, x, y);
            for a in 0..ATTRIBUTE_COUNT {
                let err = (gpu_v[a] - cpu_v[a]).abs();
                assert!(
                    err <= 2e-3 + 1e-3 * cpu_v[a].abs(),
                    "tile {tile} texel ({x},{y}) {:?}: gpu {} cpu {}",
                    Attribute::ALL[a],
                    gpu_v[a],
                    cpu_v[a]
                );
                worst[0] = worst[0].max(err);
            }
            // Pass 2 from the planes the GPU wrote (so only the resolve is compared).
            let cpu_r = resolve_texel(gpu_v, &inputs, &model, &zones);
            for c in 0..4 {
                let albedo = resolved[0][i * 4 + c];
                assert!(
                    (albedo - cpu_r.albedo[c]).abs() < 3e-3,
                    "albedo {c} ({x},{y}): {albedo} vs {}",
                    cpu_r.albedo[c]
                );
                worst[1] = worst[1].max((albedo - cpu_r.albedo[c]).abs());
                let scatter = resolved[1][i * 4 + c];
                assert!(
                    (scatter - cpu_r.scatter[c]).abs() < 5e-3 + 5e-3 * cpu_r.scatter[c].abs(),
                    "scatter {c}: {scatter} vs {}",
                    cpu_r.scatter[c]
                );
                worst[2] = worst[2].max((scatter - cpu_r.scatter[c]).abs());
                let detail = resolved[2][i * 4 + c];
                assert!((detail - cpu_r.detail[c]).abs() < 2e-3 + 1e-3 * cpu_r.detail[c].abs(), "detail {c}");
                let flush = resolved[3][i * 4 + c];
                assert!((flush - cpu_r.flush[c]).abs() < 3e-3, "flush {c} ({x},{y}): {flush} vs {}", cpu_r.flush[c]);
                worst[3] = worst[3].max((flush - cpu_r.flush[c]).abs());
                let spec = f32::from(specular[i * 4 + c]) / 255.0;
                assert!((spec - cpu_r.specular[c]).abs() <= 1.0 / 255.0 + 1e-4, "specular {c}");
                worst[4] = worst[4].max((spec - cpu_r.specular[c]).abs());
            }
        }
    }
    eprintln!(
        "max |gpu - cpu|: planes {:.2e}, albedo {:.2e}, scatter {:.2e}, flush {:.2e}, specular {:.2e}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
}

#[test]
fn skn02_a_stroke_re_resolves_only_its_rectangle() {
    // [SKN-02] "Painting a stroke re-resolves only the touched texel rectangle": after a full
    // resolve, a stroke on the moles layer resolves exactly its rectangle; texels outside keep
    // their values bit for bit, texels inside change.
    let gpu = gpu();
    let mut scene = scene();
    let mut compositor = compositor(&scene);
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    compositor.resolve(gpu, &mut encoder, &mut scene.stack, None).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let before = read_layer(gpu, compositor.resolved(), 0);
    let stroke = Rect::new(20, 12, 6, 5);
    let touched = scene
        .stack
        .paint(LayerId(2), 0, stroke, |_, _, v, c| {
            v[0] = 1.0;
            *c = 1.0;
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(touched, stroke);
    compositor.upload_rect(gpu, &scene.stack, LayerId(2), 0, touched).unwrap_or_else(|e| panic!("{e}"));
    compositor.sync_layers(gpu, &scene.stack).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let report = compositor.resolve(gpu, &mut encoder, &mut scene.stack, None).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    assert_eq!((report.rects, report.texels), (1, stroke.area()));
    let after = read_layer(gpu, compositor.resolved(), 0);
    let mut changed_inside = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = ((y * SIZE + x) * 8) as usize;
            let same = before[i..i + 8] == after[i..i + 8];
            if stroke.contains(Rect::new(x, y, 1, 1)) {
                changed_inside += usize::from(!same);
            } else {
                assert!(same, "texel ({x},{y}) outside the stroke changed");
            }
        }
    }
    assert!(changed_inside > 20, "the stroke darkened its texels ({changed_inside})");
    // Nothing dirty: nothing dispatched.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let report = compositor.resolve(gpu, &mut encoder, &mut scene.stack, None).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.texels, 0);
    submit(gpu, encoder);
}

/// GPU nanoseconds of one resolve of `rect` on tile 0 (timestamps around the pass).
fn timed_resolve(compositor: &mut SkinCompositor, stack: &mut LayerStack, rect: Rect, queries: &wgpu::QuerySet) -> f64 {
    let gpu = gpu();
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps read"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    stack.dirty_mut().mark(0, rect);
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let writes = wgpu::ComputePassTimestampWrites {
        query_set: queries,
        beginning_of_pass_write_index: Some(0),
        end_of_pass_write_index: Some(1),
    };
    compositor.resolve(gpu, &mut encoder, stack, Some(writes)).unwrap_or_else(|e| panic!("{e}"));
    encoder.resolve_query_set(queries, 0..2, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
    gpu.queue.submit([encoder.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap_or_else(|e| panic!("{e}"));
    let bytes = read.slice(..).get_mapped_range().unwrap_or_else(|e| panic!("{e}")).to_vec();
    let t0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8]));
    let t1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap_or([0; 8]));
    t1.saturating_sub(t0) as f64 * f64::from(gpu.queue.get_timestamp_period())
}

#[test]
fn ac_wp13_3_partial_rect_cost_scales_with_area() {
    // [AC-WP13-3] Re-resolve cost of a dirty rectangle on a 2048² tile with six layers, measured
    // with GPU timestamps around the pass: the time per texel stays flat from 256² up, so the cost
    // follows the rectangle's area, and a small stroke costs a tiny fraction of the tile.
    //
    // The device is shared with the other tests of this binary, which run on other threads: work
    // they submit while a resolve runs stretches its timestamps (with two test threads the 2048²
    // rectangle read 2.05x and 2.39x the per-texel cost of 256², alone about 1x). Other work only
    // ever adds time, so each size's cost is the fastest of ROUNDS measurements, taken in rounds
    // that visit every size in turn (a busy stretch of the device then slows a round, not one
    // size); the other timing tests of the crate wait on `gpu_timing_lock`.
    const ROUNDS: usize = 15;
    const SIDES: [u32; 7] = [32, 64, 128, 256, 512, 1024, 2048];
    let _timing = crate::tests::gpu_timing_lock();
    let gpu = gpu();
    assert!(gpu.has(Capability::TimestampQueries), "the test adapter supports timestamp queries");
    let size = 2048;
    let mut stack = LayerStack::new(1, size).unwrap_or_else(|e| panic!("{e}"));
    let target = Target { tile: 0, rect: Rect::new(0, 0, 1024, 1024), texel_mm: 0.1, mask: None };
    for (i, kind) in [GeneratorKind::Freckles, GeneratorKind::Moles, GeneratorKind::Veins, GeneratorKind::Acne]
        .into_iter()
        .enumerate()
    {
        let layer = generate(&GeneratorParams::new(kind), &target, LayerId(i as u32 + 1), 9, None)
            .unwrap_or_else(|e| panic!("{e}"));
        stack.push(layer).unwrap_or_else(|e| panic!("{e}"));
    }
    stack
        .push(Layer::fill(LayerId(10), "tan", BlendMode::Add, &[(Attribute::Melanin, 0.1)]))
        .unwrap_or_else(|e| panic!("{e}"));
    stack
        .push(Layer::fill(LayerId(11), "oil", BlendMode::Mix, &[(Attribute::Oiliness, 0.7)]))
        .unwrap_or_else(|e| panic!("{e}"));
    let mut compositor = SkinCompositor::new(gpu, SkinDesc { tiles: 1, tile_size: size, atlas_size: 4096 })
        .unwrap_or_else(|e| panic!("{e}"));
    compositor.sync_layers(gpu, &stack).unwrap_or_else(|e| panic!("{e}"));
    stack.dirty_mut().clear();
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("skin"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    // Warm up: the pipelines, the caches and the clocks.
    for _ in 0..3 {
        timed_resolve(&mut compositor, &mut stack, Rect::new(0, 0, size, size), &queries);
    }
    let mut fastest = [f64::INFINITY; SIDES.len()];
    for _ in 0..ROUNDS {
        for (best, side) in fastest.iter_mut().zip(SIDES) {
            let ns = timed_resolve(&mut compositor, &mut stack, Rect::new(0, 0, side, side), &queries);
            *best = best.min(ns);
        }
    }
    let rows: Vec<(u32, f64, f64)> =
        SIDES.iter().zip(fastest).map(|(&side, ns)| (side, ns, ns / f64::from(side * side))).collect();
    for (side, ns, per) in &rows {
        eprintln!("rect {side:>4}²: {:>9.3} ms, {per:.3} ns/texel", ns / 1e6);
    }
    let per = |side: u32| rows.iter().find(|r| r.0 == side).map_or(0.0, |r| r.2);
    let ms = |side: u32| rows.iter().find(|r| r.0 == side).map_or(0.0, |r| r.1);
    for side in [512, 1024, 2048] {
        let ratio = per(side) / per(256);
        eprintln!("{side}²: per-texel cost {ratio:.2}x that of 256²");
        assert!((0.5..2.0).contains(&ratio), "{side}²: per-texel cost {ratio:.2}x that of 256²");
    }
    assert!(ms(2048) > 8.0 * ms(256), "the full tile costs far more than a 256² rectangle");
    assert!(ms(64) < ms(2048) / 50.0, "a 64² stroke costs a tiny fraction of the full tile");
}

#[test]
fn skn20_gpu_exemplar_matches_the_cpu_reference() {
    // [SKN-20] The GPU generator evaluates the same function as the CPU reference.
    let gpu = gpu();
    let generator = MicroDetailGpu::new(gpu).unwrap_or_else(|e| panic!("{e}"));
    let params = MicroDetailParams { size: 128, age: 0.6, ..MicroDetailParams::default() };
    let plan = ExemplarPlan::new(&params).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let exemplar = generator.generate(gpu, &mut encoder, &plan).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let gpu_h: Vec<f32> =
        read_layer(gpu, &exemplar.height, 0).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let cpu_h = plan.heights();
    let range = cpu_h.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let mut errors: Vec<f32> = gpu_h.iter().zip(&cpu_h).map(|(a, b)| (a - b).abs()).collect();
    errors.sort_by(f32::total_cmp);
    let p999 = errors[errors.len() * 999 / 1000];
    eprintln!(
        "exemplar |gpu - cpu|: p99.9 {p999:.2e} mm, max {:.2e} mm, range {range:.3} mm",
        errors[errors.len() - 1]
    );
    assert!(p999 < 1e-4 * range.max(1e-3) * 10.0, "p99.9 {p999}");
    // The derived detail: normal x of a pit wall points away from the pit centre.
    let detail = halves(&read_layer(gpu, &exemplar.detail, 0));
    assert!(detail.iter().all(|v| v.is_finite()));
    let cavity_min = detail.chunks(4).map(|t| t[3]).fold(1.0_f32, f32::min);
    assert!(cavity_min < 0.9, "pits have cavity: {cavity_min}");
    assert_eq!(generator.late_pipelines(), 0);
    write_preview("microdetail_height.png", &cpu_h, 128);
}

#[test]
fn skn20_gpu_anti_tiling_matches_the_cpu_reference() {
    // [SKN-20] Histogram-preserving synthesis on the device equals the CPU reference texel for
    // texel (nearest fetches and the same hash: identical except at float ties).
    let gpu = gpu();
    let generator = MicroDetailGpu::new(gpu).unwrap_or_else(|e| panic!("{e}"));
    let params = MicroDetailParams { size: 64, ..MicroDetailParams::default() };
    let plan = ExemplarPlan::new(&params).unwrap_or_else(|e| panic!("{e}"));
    let gaussian = GaussianExemplar::new(&plan.heights(), 64).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let exemplar = generator.generate(gpu, &mut encoder, &plan).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let source = MicroDetailGpu::upload_gaussian(gpu, &gaussian).unwrap_or_else(|e| panic!("{e}"));
    let out = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("synthesis"),
        size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    generator.synthesize(gpu, &mut encoder, &exemplar, &plan, &source, &out, 4.0, 77).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let gpu_v: Vec<f32> = read_layer(gpu, &out, 0).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut equal = 0;
    for y in 0..256 {
        for x in 0..256 {
            let (u, v) = ((x as f32 + 0.5) / 256.0 * 4.0, (y as f32 + 0.5) / 256.0 * 4.0);
            if gaussian.sample_tiled(u, v, 77) == gpu_v[y * 256 + x] {
                equal += 1;
            }
        }
    }
    let share = f64::from(equal) / 65536.0;
    eprintln!("anti-tiling GPU == CPU for {:.3} % of texels", share * 100.0);
    assert!(share > 0.99, "{share}");
    write_preview("antitile_synthesis.png", &gpu_v, 256);
}

/// A deterministic page pattern: every byte encodes the UV tile, mip and page.
struct Pattern;

impl TileSource for Pattern {
    fn fill(&self, uv_tile: u32, mip: u8, x: u16, y: u16, out: &mut [u8]) -> Result<(), GpuError> {
        for (i, texel) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *texel = [uv_tile as u8 * 50 + mip, x as u8, y as u8, (i % 251) as u8];
        }
        Ok(())
    }
}

#[test]
fn ac_wp13_3_streaming_keeps_visible_face_tiles_at_full_mip() {
    // [SKN-03] [AC-WP13-3] Four 1024² UV tiles (face = tile 0), 128² pages, a pool of 96 pages.
    // A close-up of half the face wants mip 0 of 16 pages and the body only its coarsest levels:
    // within a few frames every visible face page is resident at mip 0 with the right texels.
    // Pulling back drops the face to a coarser mip and the pool keeps working within its budget.
    let gpu = gpu();
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut streaming = SkinStreaming::new(&mut resources, 4, 1024, 128, 8, 2, wgpu::TextureFormat::Rgba8Unorm)
        .unwrap_or_else(|e| panic!("{e}"));
    let face = TileView { uv_tile: 0, visible: Rect::new(256, 256, 512, 512), pixels_across: 2400.0 };
    let body: Vec<TileView> =
        (1..4).map(|t| TileView { uv_tile: t, visible: Rect::new(0, 0, 1024, 1024), pixels_across: 150.0 }).collect();
    assert_eq!(streaming.wanted_mip(&face), 0);
    assert_eq!(streaming.wanted_mip(&body[0]), 2);
    let mut frames = 0;
    while !streaming.is_view_resident(&face) || !body.iter().all(|b| streaming.is_view_resident(b)) {
        frames += 1;
        assert!(frames < 20, "the close-up never became resident");
        streaming.begin_frame();
        streaming.request_view(&face).unwrap_or_else(|e| panic!("{e}"));
        for b in &body {
            streaming.request_view(b).unwrap_or_else(|e| panic!("{e}"));
        }
        streaming.update(&mut resources, 32, &Pattern).unwrap_or_else(|e| panic!("{e}"));
    }
    eprintln!("face close-up resident at mip 0 after {frames} frames, {} pages loaded", streaming.pool().stats().loads);
    // The resident face page holds its texels: read its slot back from the pool.
    let texture = streaming.texture_of(0).unwrap_or(0);
    let entry =
        streaming.pool().lookup(TileRequest { texture, mip: 0, x: 3, y: 3 }).unwrap_or_else(|| panic!("resident"));
    assert_eq!(entry.mip, 0);
    let location = streaming.pool().slot_location(entry.slot);
    let pool = resources.texture(streaming.pool().pool_texture()).unwrap_or_else(|| panic!("pool texture")).raw();
    let layer = read_layer(gpu, pool, location.layer);
    let side = pool.width();
    let at = ((location.y * side + location.x) * 4) as usize;
    assert_eq!(&layer[at..at + 3], &[0, 3, 3], "page (3, 3) of mip 0 of the face");
    // Pull back: the face now wants mip 2; it becomes resident and old pages may be evicted.
    let far = TileView { pixels_across: 300.0, ..face };
    assert_eq!(streaming.wanted_mip(&far), 1);
    for _ in 0..4 {
        streaming.begin_frame();
        streaming.request_view(&far).unwrap_or_else(|e| panic!("{e}"));
        streaming.update(&mut resources, 32, &Pattern).unwrap_or_else(|e| panic!("{e}"));
    }
    assert!(streaming.is_view_resident(&far));
    assert!(streaming.pool().resident() <= streaming.pool().slot_count());
}

/// Writes a grey preview of a float field (normalised) for a human look.
fn write_preview(name: &str, values: &[f32], size: usize) {
    let (lo, hi) = values.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (hi - lo).max(1e-9);
    let bytes: Vec<u8> = values.iter().map(|&v| ((v - lo) / span * 255.0).round() as u8).collect();
    write_png(name, Image::new(Format::Png, size as u32, size as u32, 1, ColorSpace::Data, Samples::U8(bytes)));
}

fn write_png(name: &str, image: humanart_io::Result<Image>) {
    let image = image.unwrap_or_else(|e| panic!("{e}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin/previews");
    let _ = std::fs::create_dir_all(&dir);
    let png = humanart_io::png::encode(&image).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(dir.join(name), png).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn skn10_preview_of_features_and_a_melanin_sweep() {
    // A picture to look at: one 256² tile with a melanin sweep from left (0) to right (1) as a
    // content layer, freckles, moles, veins and a scar, resolved on the GPU and written as sRGB.
    let gpu = gpu();
    let size = 256;
    let mut stack = LayerStack::new(1, size).unwrap_or_else(|e| panic!("{e}"));
    let mut sweep = Layer::content(LayerId(1), "sweep", BlendMode::Mix, [Some(Attribute::Melanin), None, None, None]);
    let mut c = LayerContent::new(0, Rect::new(0, 0, size, size)).unwrap_or_else(|e| panic!("{e}"));
    for (i, (v, cov)) in c.values.iter_mut().zip(c.coverage.iter_mut()).enumerate() {
        v[0] = q((i as u32 % size) as f32 / (size - 1) as f32);
        *cov = 1.0;
    }
    if let LayerKind::Content { tiles, .. } = &mut sweep.kind {
        tiles.push(c);
    }
    stack.push(sweep).unwrap_or_else(|e| panic!("{e}"));
    let target = Target { tile: 0, rect: Rect::new(0, 0, size, size), texel_mm: 0.25, mask: None };
    for (i, kind) in [GeneratorKind::Freckles, GeneratorKind::Moles, GeneratorKind::Veins, GeneratorKind::Scars]
        .into_iter()
        .enumerate()
    {
        let layer = generate(&GeneratorParams::new(kind), &target, LayerId(i as u32 + 2), 21, None)
            .unwrap_or_else(|e| panic!("{e}"));
        stack.push(layer).unwrap_or_else(|e| panic!("{e}"));
    }
    let mut compositor = SkinCompositor::new(gpu, SkinDesc { tiles: 1, tile_size: size, atlas_size: 1024 })
        .unwrap_or_else(|e| panic!("{e}"));
    compositor.sync_layers(gpu, &stack).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    compositor.resolve(gpu, &mut encoder, &mut stack, None).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let albedo = halves(&read_layer(gpu, compositor.resolved(), 0));
    let mut bytes = Vec::with_capacity((size * size * 3) as usize);
    for t in albedo.chunks(4) {
        for c in &t[..3] {
            bytes.push(humanart_math::color::transfer::linear_to_srgb8(*c));
        }
    }
    // The sweep darkens monotonically along a row away from features (row 2, sampled every 32).
    let lum = |x: usize| albedo[(2 * size as usize + x) * 4 + 1];
    assert!(lum(16) > lum(128) && lum(128) > lum(240), "{} {} {}", lum(16), lum(128), lum(240));
    write_png("skin_features_sweep.png", Image::new(Format::Png, size, size, 3, ColorSpace::Srgb, Samples::U8(bytes)));
    let _ = f32_to_f16_bits(0.0);
}

#[test]
fn perf_r1_prepare_allocates_nothing() {
    // [PERF-R1] The CPU half of a resolve (draining dirty rectangles, filling the parameter
    // mirror) allocates nothing in steady state, strokes and whole-set invalidations alike. The
    // counting allocator is the test binary's (installed in `deform/tests.rs`).
    assert!(humanart_core::alloc::is_installed());
    let mut scene = scene();
    let mut compositor = compositor(&scene);
    let stats = humanart_core::alloc::count_in(|| {
        for i in 0..50_u32 {
            scene.stack.dirty_mut().mark(0, Rect::new(i % 40, i % 30, 9, 7));
            scene.stack.dirty_mut().mark(1, Rect::new(3, i % 50, 12, 4));
            if i % 10 == 0 {
                scene.stack.dirty_mut().mark_all();
            }
            let report = compositor.prepare(&mut scene.stack).unwrap_or_else(|e| panic!("{e}"));
            assert!(report.rects > 0);
        }
    });
    assert_eq!(stats.events(), 0, "{stats:?}");
}

/// Reads a whole `COPY_SRC` buffer.
fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skin buffer readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap_or_else(|e| panic!("{e}"));
    rx.recv().unwrap_or_else(|e| panic!("{e}")).unwrap_or_else(|e| panic!("{e}"));
    let out = staging.slice(..).get_mapped_range().unwrap_or_else(|e| panic!("{e}")).to_vec();
    staging.unmap();
    out
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// Drivers over four regions (one driver points past the strain buffer), their table, the
/// per-region strain buffer and a wrinkle set.
fn wrinkle_setup() -> (DriverTable, super::wrinkle::WrinkleGpu, wgpu::Buffer, [f32; 4], [f32; 4]) {
    let gpu = gpu();
    let driver = |i: u32, region: u16, expressions: Vec<(u32, f32)>, c: f32, s: f32| WrinkleDriver {
        name: format!("driver-{i}"),
        region,
        expressions,
        compression_gain: c,
        stretch_gain: s,
    };
    let drivers = vec![
        driver(0, 1, vec![(0, 0.5)], 8.0, 6.0),
        driver(1, 2, vec![(1, 1.0), (0, -0.2)], 40.0, 0.0),
        driver(2, 0, vec![], -3.0, 40.0),
        driver(3, 3, vec![(2, 0.7)], 2.0, 2.0),
        driver(4, 7, vec![(2, 0.4)], 1.0, 1.0),
    ];
    let table = DriverTable::new(&drivers).unwrap_or_else(|e| panic!("{e}"));
    let stretch: [f32; 4] = [0.03, 0.2, 0.05, f32::NAN];
    let compression: [f32; 4] = [0.02, 0.05, 0.03, 0.4];
    let strain_bytes: Vec<u8> =
        stretch.iter().zip(&compression).flat_map(|(s, c)| [s.to_le_bytes(), c.to_le_bytes()]).flatten().collect();
    let strain = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test region strain"),
        size: strain_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&strain, 0, &strain_bytes);
    let desc = super::wrinkle::WrinkleDesc { size: 64, tiles: 2, drivers: 8 };
    let mut wrinkles = super::wrinkle::WrinkleGpu::new(gpu, desc).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.set_drivers(gpu, &table).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.bind_strain(gpu, &strain, 4).unwrap_or_else(|e| panic!("{e}"));
    (table, wrinkles, strain, stretch, compression)
}

#[test]
fn skn30_gpu_activations_match_the_cpu_drivers() {
    // [SKN-30] [EVAL-06]: fold and stretch per driver from the expression terms and the region
    // strain buffer on the GPU equal DriverTable::activate (itself equal to WrinkleDrivers::evaluate).
    let gpu = gpu();
    let (table, mut wrinkles, _strain, stretch, compression) = wrinkle_setup();
    let weights = [0.8, 0.3, 0.9];
    let mut terms = [0.0; 5];
    table.expression_terms(&weights, &mut terms).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.set_terms(gpu, &terms).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    wrinkles.record_activate(&mut encoder).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let got = floats(&read_buffer(gpu, wrinkles.activations()));
    let (mut fold, mut st) = ([0.0; 5], [0.0; 5]);
    table.activate(&terms, &stretch, &compression, &mut fold, &mut st).unwrap_or_else(|e| panic!("{e}"));
    for d in 0..5 {
        assert!((got[2 * d] - fold[d]).abs() < 1e-6, "fold {d}: {} vs {}", got[2 * d], fold[d]);
        assert!((got[2 * d + 1] - st[d]).abs() < 1e-6, "stretch {d}: {} vs {}", got[2 * d + 1], st[d]);
    }
    // Saturation, a negative gain, a NaN strain and a region beyond the buffer are all covered.
    assert_eq!(fold[1], 1.0);
    assert_eq!(st[2], 1.0);
    assert_eq!(st[3], 0.0);
    assert_eq!(fold[4], terms[4].clamp(0.0, 1.0));
    assert_eq!(wrinkles.late_pipelines(), 0);
}

#[test]
fn skn30_gpu_blend_matches_the_cpu_twin() {
    // [SKN-30] [EXP-04]: the baked maps on the GPU (rgba16float) blended by the GPU activations
    // equal blend_texel on the same f16-rounded texels.
    let gpu = gpu();
    let (table, mut wrinkles, _strain, stretch, compression) = wrinkle_setup();
    let mut rng = Rng::new(0x530);
    let mut maps = Vec::new();
    for driver in 0..5_u16 {
        let (x, y) = (rng.index(24) as u32, rng.index(24) as u32);
        let rect = Rect::new(x, y, 20 + rng.index(20) as u32, 16 + rng.index(20) as u32);
        let n = (rect.w * rect.h) as usize;
        let height: Vec<f32> = (0..n).map(|_| -0.3 * rng.f32()).collect();
        let cavity = height.iter().map(|h| 0.8 * h).collect();
        maps.push(WrinkleMap { driver, tile: 5, rect, height, cavity, flatten: 0.2 * f32::from(driver) });
    }
    let baked = WrinkleMaps::bake(64, &[5, 9], &[0.4, 0.4], &maps).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.upload_maps(gpu, &baked).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(wrinkles.tile_slot(5), Some(0));
    assert_eq!(wrinkles.tile_slot(9), Some(1));
    assert_eq!(wrinkles.tile_slot(0), None);
    let mut terms = [0.0; 5];
    table.expression_terms(&[0.6, 0.2, 1.0], &mut terms).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.set_terms(gpu, &terms).unwrap_or_else(|e| panic!("{e}"));
    let rect = Rect::new(0, 0, 64, 64);
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test wrinkle blend"),
        size: 64 * 64 * super::wrinkle::BLEND_TEXEL_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    wrinkles.record_activate(&mut encoder).unwrap_or_else(|e| panic!("{e}"));
    wrinkles.blend_rect(gpu, &mut encoder, 0, rect, &out).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let got = floats(&read_buffer(gpu, &out));
    let (mut fold, mut st) = ([0.0; 5], [0.0; 5]);
    table.activate(&terms, &stretch, &compression, &mut fold, &mut st).unwrap_or_else(|e| panic!("{e}"));
    let h = |x: f32| f16_bits_to_f32(f32_to_f16_bits(x));
    let (mut worst, mut nonzero) = (0.0_f32, 0);
    for y in 0..64 {
        for x in 0..64 {
            let t = baked.texel(5, x, y).unwrap_or_else(|| panic!("texel"));
            let mut r = *t;
            for k in 0..2 {
                r.height[k] = h(r.height[k]);
                r.slope[k] = [h(r.slope[k][0]), h(r.slope[k][1])];
                r.cavity[k] = h(r.cavity[k]);
                r.flatten[k] = h(r.flatten[k]);
            }
            let want = blend_texel(&r, &fold, &st);
            let i = ((y * 64 + x) * 8) as usize;
            let g = &got[i..i + 8];
            let diffs = [
                g[0] - want.height,
                g[1] - want.slope[0],
                g[2] - want.slope[1],
                g[3] - want.cavity,
                g[4] - want.detail_scale,
            ];
            worst = diffs.iter().fold(worst, |a, d| a.max(d.abs()));
            if want.height != 0.0 {
                nonzero += 1;
            }
        }
    }
    assert!(worst < 1e-5, "worst {worst}");
    assert!(nonzero > 500, "{nonzero} texels carry wrinkles");
    // Slot 1 (tile 9) carries nothing.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    wrinkles.blend_rect(gpu, &mut encoder, 1, Rect::new(8, 8, 16, 16), &out).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let got = floats(&read_buffer(gpu, &out));
    for t in got[..16 * 16 * 8].chunks(8) {
        assert_eq!(t[..5], [0.0, 0.0, 0.0, 0.0, 1.0]);
    }
    // Input checks.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(wrinkles.blend_rect(gpu, &mut encoder, 2, rect, &out).is_err());
    assert!(wrinkles.blend_rect(gpu, &mut encoder, 0, Rect::new(60, 0, 8, 8), &out).is_err());
    assert!(wrinkles.set_terms(gpu, &[0.0; 2]).is_err());
    let small = WrinkleMaps::bake(32, &[5], &[0.4], &[]).unwrap_or_else(|e| panic!("{e}"));
    assert!(wrinkles.upload_maps(gpu, &small).is_err());
}

/// The CPU box average of page `(px, py)` at `mip` of one layer of `texels` (RGBA, `size` wide).
fn box_page(texels: &[f32], size: u32, page: u32, mip: u8, px: u32, py: u32) -> Vec<[f32; 4]> {
    let s = 1_u32 << mip;
    let mut out = Vec::new();
    for y in 0..page {
        for x in 0..page {
            let mut sum = [0.0_f32; 4];
            for j in 0..s {
                for i in 0..s {
                    let qx = ((px * page + x) * s + i).min(size - 1);
                    let qy = ((py * page + y) * s + j).min(size - 1);
                    let at = ((qy * size + qx) * 4) as usize;
                    for c in 0..4 {
                        sum[c] += texels[at + c];
                    }
                }
            }
            out.push(sum.map(|v| v / (s * s) as f32));
        }
    }
    out
}

/// Reads the `page²` texels of a pool slot (RGBA16F) back.
fn read_slot(gpu: &Gpu, pool: &wgpu::Texture, at: [u32; 3], page: u32) -> Vec<f32> {
    let layer = read_layer(gpu, pool, at[2]);
    let row = pool.width() as usize * 8;
    let mut out = Vec::new();
    for y in 0..page as usize {
        let start = (at[1] as usize + y) * row + at[0] as usize * 8;
        out.extend(halves(&layer[start..start + page as usize * 8]));
    }
    out
}

#[test]
fn skn03_gpu_to_pool_copy_streams_the_resolved_set() {
    // [SKN-03] The editor's resolved set streams into the pool without a readback: after
    // `update_from_gpu`, every resident page's slot holds the resolved texels of its square,
    // box-averaged to its mip (mip 0 bit-exact), for the albedo and the flush channel, and the
    // specular (rgba8unorm) array streams the same way.
    let gpu = gpu();
    let mut scene = scene();
    let mut compositor = compositor(&scene);
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    compositor.resolve(gpu, &mut encoder, &mut scene.stack, None).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    let page = 16;
    for (channel, layers_per_tile) in [(0_u32, 4_u32), (3, 4)] {
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut streaming = SkinStreaming::new(&mut resources, 2, SIZE, page, 4, 2, wgpu::TextureFormat::Rgba16Float)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut pages =
            super::page_copy::ResolvedPages::new(gpu, compositor.resolved(), layers_per_tile, channel, page)
                .unwrap_or_else(|e| panic!("{e}"));
        let near = TileView { uv_tile: 1, visible: Rect::new(0, 0, SIZE, SIZE), pixels_across: 4.0 * SIZE as f32 };
        let far = TileView { uv_tile: 0, visible: Rect::new(0, 0, SIZE, SIZE), pixels_across: 20.0 };
        assert_eq!(streaming.wanted_mip(&near), 0);
        assert_eq!(streaming.wanted_mip(&far), 1);
        let mut frames = 0;
        let uploaded_before = resources.stats().bytes_uploaded;
        let (mut loaded, mut table_bytes) = (0_u64, 0_u64);
        while !streaming.is_view_resident(&near) || !streaming.is_view_resident(&far) {
            frames += 1;
            assert!(frames < 10, "never resident");
            streaming.begin_frame();
            streaming.request_view(&near).unwrap_or_else(|e| panic!("{e}"));
            streaming.request_view(&far).unwrap_or_else(|e| panic!("{e}"));
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let report = streaming
                .update_from_gpu(gpu, &mut resources, &mut encoder, 8, &mut pages)
                .unwrap_or_else(|e| panic!("{e}"));
            loaded += u64::from(report.loaded);
            table_bytes += report.table_bytes;
            submit(gpu, encoder);
        }
        assert_eq!(pages.late_pipelines(), 0);
        // (todo 6) The pool uploads no texels for a page the GPU copies: only page-table rows went
        // to the queue. Before, every such page also cost one upload of zeros of its size.
        let uploaded = resources.stats().bytes_uploaded - uploaded_before;
        let page_bytes = u64::from(page * page) * 8;
        assert!(loaded >= 16 + 4, "{loaded} pages loaded");
        assert!(table_bytes > 0 && table_bytes < page_bytes, "{table_bytes} page-table bytes");
        assert_eq!(uploaded, table_bytes, "{loaded} pages streamed: only page-table rows are uploaded");
        eprintln!(
            "channel {channel}: {loaded} pages streamed, {uploaded} B uploaded (page table); zero pages would add {} B",
            loaded * page_bytes
        );
        let pool_texture =
            resources.texture(streaming.pool().pool_texture()).unwrap_or_else(|| panic!("pool")).raw().clone();
        let mut worst = 0.0_f32;
        let mut checked = 0;
        for (view, mip) in [(near, 0_u8), (far, 1)] {
            let source = halves(&read_layer(gpu, compositor.resolved(), view.uv_tile * layers_per_tile + channel));
            let texture = streaming.texture_of(view.uv_tile).unwrap_or(0);
            let side = SIZE / (page << mip);
            for py in 0..side {
                for px in 0..side {
                    let request = TileRequest { texture, mip, x: px as u16, y: py as u16 };
                    let entry = streaming.pool().lookup(request).unwrap_or_else(|| panic!("resident"));
                    assert_eq!(entry.mip, mip);
                    let at = streaming.pool().slot_location(entry.slot);
                    let got = read_slot(gpu, &pool_texture, [at.x, at.y, at.layer], page);
                    let want = box_page(&source, SIZE, page, mip, px, py);
                    for (g, w) in got.chunks(4).zip(&want) {
                        for c in 0..4 {
                            let d = (g[c] - w[c]).abs();
                            if mip == 0 {
                                assert_eq!(g[c], w[c], "mip 0 copies exactly");
                            }
                            // One f16 step of the averaged value.
                            assert!(d <= w[c].abs() * 1e-3 + 1e-4, "page {px},{py} mip {mip}: {} vs {}", g[c], w[c]);
                            worst = worst.max(d);
                        }
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 16 + 4);
        eprintln!("channel {channel}: {checked} pages copied on the GPU, worst |diff| {worst:e}");
    }
    // The specular array streams as rgba8unorm.
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut streaming = SkinStreaming::new(&mut resources, 2, SIZE, page, 4, 2, wgpu::TextureFormat::Rgba8Unorm)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut pages =
        super::page_copy::ResolvedPages::new(gpu, compositor.specular(), 1, 0, page).unwrap_or_else(|e| panic!("{e}"));
    let view = TileView { uv_tile: 0, visible: Rect::new(16, 16, 16, 16), pixels_across: 4.0 * SIZE as f32 };
    streaming.begin_frame();
    streaming.request_view(&view).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let report =
        streaming.update_from_gpu(gpu, &mut resources, &mut encoder, 8, &mut pages).unwrap_or_else(|e| panic!("{e}"));
    submit(gpu, encoder);
    assert!(report.loaded >= 1);
    let texture = streaming.texture_of(0).unwrap_or(0);
    let entry =
        streaming.pool().lookup(TileRequest { texture, mip: 0, x: 1, y: 1 }).unwrap_or_else(|| panic!("resident"));
    let at = streaming.pool().slot_location(entry.slot);
    let pool_texture = resources.texture(streaming.pool().pool_texture()).unwrap_or_else(|| panic!("pool")).raw();
    let pool_layer = read_layer(gpu, pool_texture, at.layer);
    let source = read_layer(gpu, compositor.specular(), 0);
    let row = pool_texture.width() as usize * 4;
    for y in 0..page as usize {
        let got = &pool_layer[(at.y as usize + y) * row + at.x as usize * 4..][..page as usize * 4];
        let want = &source[((16 + y) * SIZE as usize + 16) * 4..][..page as usize * 4];
        assert_eq!(got, want, "specular row {y}");
    }
    // Mismatches are refused.
    let mut wrong =
        super::page_copy::ResolvedPages::new(gpu, compositor.resolved(), 4, 0, page).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(streaming.update_from_gpu(gpu, &mut resources, &mut encoder, 8, &mut wrong).is_err());
    assert!(super::page_copy::ResolvedPages::new(gpu, compositor.resolved(), 4, 4, page).is_err());
    assert!(super::page_copy::ResolvedPages::new(gpu, compositor.resolved(), 4, 0, 24).is_err());
    assert!(super::page_copy::ResolvedPages::new(gpu, compositor.planes(), 5, 0, page).is_ok());
}
