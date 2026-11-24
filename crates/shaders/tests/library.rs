//! `[AC-WP05-2]` The shared WGSL libraries run on the real GPU and are compared with their CPU
//! twins and with closed-form results:
//!
//! - packing (`lib/packing.wgsl`): bit for bit against `humanart_math::pack`;
//! - colour (`lib/color.wgsl`): the sRGB curve, luminance, exposure and every tone mapper against
//!   `humanart_math::color`, including the tone-mapper numbering, names and constants;
//! - math (`lib/math.wgsl`): quaternions against `humanart_math::Quat`, the orthonormal basis
//!   against `DVec3::any_orthonormal_pair`;
//! - BRDF (`lib/brdf.wgsl`): the white furnace (Lambert, the GGX normalisation, the single-scatter
//!   albedo bounded by 1 and equal to an f64 CPU quadrature of the same integrand), the split-sum
//!   DFG against that albedo for every override combination, Fresnel and masking against f64;
//! - sampling (`lib/sampling.wgsl`): the sequences bit for bit, and the moments of every warp
//!   against their closed forms or an f64 quadrature of the density the library claims.
//!
//! Every kernel is a compute entry point of a `test/*.wgsl` shader sharing the bindings of
//! `test/harness.wgsl`: a `Params` uniform, an input and an output array of 4-word elements.

use humanart_core::rng::Rng;
use humanart_gpu::Gpu;
use humanart_math::color::{spaces, tonemap, transfer};
use humanart_math::pack;
use humanart_math::{DVec3, Quat, Vec2, Vec3, Vec4};
use humanart_shaders::{manifest, preprocess};

// ---------------------------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------------------------

/// One device for the whole test binary.
fn gpu() -> &'static Gpu {
    // The process's shared device (ADR-040).
    humanart_gpu::shared_headless().unwrap_or_else(|e| panic!("this machine has a GPU: {e}"))
}

/// One dispatch of a test kernel.
struct Run<'a> {
    shader: &'static str,
    entry: &'static str,
    constants: &'a [(&'a str, f64)],
    count: u32,
    mode: u32,
    a: f32,
    b: f32,
    input: &'a [[u32; 4]],
    output_len: usize,
}

impl Run<'_> {
    const fn new(shader: &'static str, entry: &'static str) -> Self {
        Run { shader, entry, constants: &[], count: 0, mode: 0, a: 0.0, b: 0.0, input: &[], output_len: 0 }
    }
}

/// Runs `run` on the device and reads the output array back.
fn dispatch(run: &Run<'_>) -> Vec<[u32; 4]> {
    let gpu = gpu();
    let device = &gpu.device;
    assert!(manifest::shader(run.shader).is_some_and(|s| s.entry_point(run.entry).is_some()), "{}", run.entry);
    let composed = preprocess::compose(run.shader, &[]).unwrap_or_else(|e| panic!("{e}"));
    let module = humanart_gpu::error::capture(device, "test module", || {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(run.shader),
            source: wgpu::ShaderSource::Wgsl(composed.source().into()),
        })
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let storage = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry =
        |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("harness"),
        entries: &[
            entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            entry(1, storage(true)),
            entry(2, storage(false)),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("harness"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = humanart_gpu::error::capture(device, "test pipeline", || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(run.entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(run.entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: run.constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        })
    })
    .unwrap_or_else(|e| panic!("{e}"));

    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&run.count.to_le_bytes());
    params.extend_from_slice(&run.mode.to_le_bytes());
    params.extend_from_slice(&run.a.to_le_bytes());
    params.extend_from_slice(&run.b.to_le_bytes());
    let words = |elements: &[[u32; 4]]| -> Vec<u8> {
        let mut bytes: Vec<u8> = elements.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        if bytes.is_empty() {
            bytes.resize(16, 0);
        }
        bytes
    };
    let buffer = |label, contents: &[u8], usage| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, contents);
        buffer
    };
    let uniform = buffer("params", &params, wgpu::BufferUsages::UNIFORM);
    let input = buffer("input", &words(run.input), wgpu::BufferUsages::STORAGE);
    let output_size = (run.output_len.max(1) * 16) as u64;
    let output =
        buffer("output", &vec![0; output_size as usize], wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("harness"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });
    let workgroup = run.constants.iter().find(|(n, _)| *n == "WORKGROUP_SIZE").map_or(64, |(_, v)| *v as u32);
    let groups = run.count.div_ceil(workgroup);
    assert!(groups <= gpu.limits().max_compute_workgroups_per_dimension, "{groups} workgroups");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("harness") });
    {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(run.entry), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
        if let Err(e) = result {
            panic!("the buffer maps: {e}");
        }
    });
    if let Err(e) = device.poll(wgpu::PollType::wait_indefinitely()) {
        panic!("the device answers: {e}");
    }
    let bytes = staging.slice(..).get_mapped_range().unwrap_or_else(|e| panic!("the buffer is mapped: {e}")).to_vec();
    staging.unmap();
    let (elements, _) = bytes.as_chunks::<16>();
    elements
        .iter()
        .take(run.output_len)
        .map(|c| {
            let w = |i: usize| u32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]);
            [w(0), w(1), w(2), w(3)]
        })
        .collect()
}

/// Four floats as an input element.
fn bits(v: [f32; 4]) -> [u32; 4] {
    v.map(f32::to_bits)
}

/// An output element as four floats.
fn floats(v: [u32; 4]) -> [f32; 4] {
    v.map(f32::from_bits)
}

/// Asserts `|gpu - cpu| <= tolerance` component-wise, naming the case.
#[track_caller]
fn assert_close(what: &str, case: impl std::fmt::Debug, gpu: &[f32], cpu: &[f32], tolerance: f32) {
    for (g, c) in gpu.iter().zip(cpu) {
        assert!((g - c).abs() <= tolerance, "{what} {case:?}: gpu {gpu:?}, cpu {cpu:?} (tolerance {tolerance})");
    }
}

/// The largest magnitude in `values`, at least 1: scales an absolute tolerance to a relative one.
fn scale(values: &[f32]) -> f32 {
    values.iter().fold(1.0, |m, v| m.max(v.abs()))
}

/// `n` random unit directions plus the axes and a few awkward ones.
fn directions(rng: &mut Rng, n: usize) -> Vec<Vec3> {
    let mut out = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(-0.0, -0.0, -1.0),
        Vec3::new(0.577_350_3, -0.577_350_3, -0.577_350_3),
    ];
    while out.len() < n {
        let v = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32());
        if let Some(u) = v.try_normalize() {
            out.push(u);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------------------------

/// An octahedral encoder's CPU word, its CPU decoder and the bits of its two fields.
type OctEncoder = (u32, fn(u32) -> Vec3, [u32; 2]);

/// The fields of a word written by a WGSL normalised-integer built-in: (the input components, bits
/// per field, signed, scale). `None`: a library encoder whose word must be exact.
type Layout = Option<(&'static [usize], u32, bool, f64)>;

/// The words of `encode` modes 1 and 2 (see test/packing.wgsl).
const LAYOUTS: [[Layout; 4]; 2] = [
    [
        None,
        Some((&[0, 1, 2, 3], 8, true, 127.0)),
        Some((&[0, 1, 2, 3], 8, false, 255.0)),
        Some((&[0, 1], 16, true, 32767.0)),
    ],
    [None, None, Some((&[0, 1], 16, false, 65535.0)), Some((&[2, 3], 16, false, 65535.0))],
];

#[test]
fn ac_wp05_2_packing_encoders_match_the_cpu() {
    let mut rng = Rng::new(0x5EED_0501);
    // Mode 0: directions (some not unit length) with a bitangent sign.
    let dirs: Vec<[f32; 4]> = directions(&mut rng, 20_000)
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            let scale = if i % 3 == 0 { rng.range_f32(0.01, 50.0) } else { 1.0 };
            [d.x * scale, d.y * scale, d.z * scale, if rng.chance(0.5) { -1.0 } else { 1.0 }]
        })
        .chain([[0.0; 4], [f32::NAN, 0.0, 1.0, 1.0], [f32::INFINITY, 0.0, 0.0, -1.0]])
        .collect();
    let input: Vec<[u32; 4]> = dirs.iter().copied().map(bits).collect();
    let n = input.len();
    let gpu = dispatch(&Run {
        count: n as u32,
        mode: 0,
        input: &input,
        output_len: n,
        ..Run::new("test/packing.wgsl", "encode")
    });
    // The octahedral encoders pick the best of four grid points by chord length. When two of them
    // are equally close within f32 rounding, the GPU (whose `length` and division are not correctly
    // rounded) may pick the other one: a neighbouring code, never a worse one. Everything else is
    // bit-exact.
    let mut ties = [0_usize; 3];
    let mut worst_tie = 0.0_f32;
    for (v, g) in dirs.iter().zip(&gpu) {
        let d = Vec3::new(v[0], v[1], v[2]);
        let snorm = pack::pack_snorm_10_10_10_2(Vec4::new(v[0], v[1], v[2], v[3]));
        assert_eq!(g[3], snorm, "snorm 10-10-10-2 of {v:?}");
        let unit = d.normalize_or(Vec3::new(0.0, 0.0, 1.0));
        let encoders: [OctEncoder; 3] = [
            (pack::pack_oct16(d), pack::unpack_oct16, [16, 16]),
            (u32::from(pack::pack_oct8(d)), |p| pack::unpack_oct8(p as u16), [8, 8]),
            (pack::pack_tangent(d, v[3]), |p| pack::unpack_tangent(p).0, [16, 15]),
        ];
        for (k, (cpu, decode, [x_bits, y_bits])) in encoders.into_iter().enumerate() {
            if g[k] == cpu {
                continue;
            }
            let field = |p: u32, shift: u32, bits: u32| {
                let raw = (p >> shift) & ((1 << bits) - 1);
                (raw << (32 - bits)) as i32 >> (32 - bits)
            };
            let neighbours = (field(g[k], 0, x_bits) - field(cpu, 0, x_bits)).abs() <= 1
                && (field(g[k], x_bits, y_bits) - field(cpu, x_bits, y_bits)).abs() <= 1
                && (g[k] ^ cpu).checked_shr(x_bits + y_bits).unwrap_or(0) == 0;
            let (chord_gpu, chord_cpu) = (decode(g[k]).distance(unit), decode(cpu).distance(unit));
            worst_tie = worst_tie.max(chord_gpu - chord_cpu);
            assert!(
                neighbours && chord_gpu <= chord_cpu + 1e-6,
                "encoder {k} of {v:?}: gpu {:#010x} (error {chord_gpu}), cpu {cpu:#010x} (error {chord_cpu})",
                g[k]
            );
            ties[k] += 1;
        }
    }
    println!(
        "octahedral near-ties resolved differently (oct16, oct8, tangent) of {n}: {ties:?}, worst extra chord {worst_tie:e}"
    );
    assert!(ties.iter().all(|&t| t * 100 < n), "near-ties are rare: {ties:?} of {n}");

    // Modes 1 and 2: values around and beyond the normalised ranges, plus exact levels.
    let mut values: Vec<[f32; 4]> = (0..20_000).map(|_| [(); 4].map(|()| rng.range_f32(-1.5, 1.5))).collect::<Vec<_>>();
    values.extend((0..=8).map(|k| {
        let x = k as f32 / 8.0;
        [x, -x, 1.0 - x, x * 0.5]
    }));
    values.extend([[0.0, -0.0, 1.0, -1.0], [65504.0, -65504.0, 1e-8, 6.1e-5], [1e5, -1e5, 0.333_333_34, 3.0]]);
    let input: Vec<[u32; 4]> = values.iter().copied().map(bits).collect();
    let n = input.len();
    // The library's own encoders (10-10-10-2, the halves) are bit-exact. The WGSL built-ins
    // pack4x8snorm / pack4x8unorm / pack2x16snorm / pack2x16unorm are the driver's: they may round
    // a value at (or within f32 rounding of) a half-way point the other way, one code apart.
    let mut rounding_ties = 0_usize;
    for mode in [1_u32, 2] {
        let gpu = dispatch(&Run {
            count: n as u32,
            mode,
            input: &input,
            output_len: n,
            ..Run::new("test/packing.wgsl", "encode")
        });
        for (v, g) in values.iter().zip(&gpu) {
            let v4 = Vec4::new(v[0], v[1], v[2], v[3]);
            let cpu = if mode == 1 {
                [
                    pack::pack_unorm_10_10_10_2(v4),
                    pack::pack4x8snorm(v4),
                    pack::pack4x8unorm(v4),
                    pack::pack2x16snorm(Vec2::new(v[0], v[1])),
                ]
            } else {
                let h = pack::pack4x16float(v4);
                [h[0], h[1], pack::pack2x16unorm(Vec2::new(v[0], v[1])), pack::pack2x16unorm(Vec2::new(v[2], v[3]))]
            };
            for (k, layout) in LAYOUTS[mode as usize - 1].iter().enumerate() {
                if g[k] == cpu[k] {
                    continue;
                }
                let Some((inputs, bits, signed, scale)) = layout else {
                    panic!("mode {mode} word {k} of {v:?}: gpu {:#010x}, cpu {:#010x}", g[k], cpu[k]);
                };
                for (field, &component) in inputs.iter().enumerate() {
                    let shift = field as u32 * bits;
                    let code = |w: u32| {
                        let raw = (w >> shift) & ((1 << bits) - 1);
                        if *signed { i64::from((raw << (32 - bits)) as i32 >> (32 - bits)) } else { i64::from(raw) }
                    };
                    if code(g[k]) == code(cpu[k]) {
                        continue;
                    }
                    let lo = if *signed { -1.0 } else { 0.0 };
                    let scaled = scale * f64::from(v[component]).clamp(lo, 1.0);
                    let from_half = (scaled - scaled.floor() - 0.5).abs();
                    assert!(
                        (code(g[k]) - code(cpu[k])).abs() == 1
                            && from_half <= 4.0 * f64::from(f32::EPSILON) * scaled.abs().max(1.0),
                        "mode {mode} word {k} of {v:?}: gpu {:#010x}, cpu {:#010x}, scaled {scaled}",
                        g[k],
                        cpu[k]
                    );
                    rounding_ties += 1;
                }
            }
        }
    }
    println!("built-in packers rounding a half-way value the other way: {rounding_ties} fields of {n} x 12");
}

#[test]
fn ac_wp05_2_packing_decoders_match_the_cpu() {
    let mut rng = Rng::new(0x5EED_0502);
    let mut words: Vec<u32> = (0..20_000).map(|_| rng.next_u32()).collect();
    words.extend([0, u32::MAX, 0x8000_8000, 0x7fff_7fff, 0x8000_0000, 0x0000_8000, 0x3ff, 0xc000_0000]);
    let input: Vec<[u32; 4]> = words.iter().map(|&w| [w, rng.next_u32(), 0, 0]).collect();
    let n = input.len();
    for mode in 0..6_u32 {
        let gpu = dispatch(&Run {
            count: n as u32,
            mode,
            input: &input,
            output_len: n,
            ..Run::new("test/packing.wgsl", "decode")
        });
        for (p, g) in input.iter().zip(&gpu) {
            let g = floats(*g);
            let cpu: [f32; 4] = match mode {
                0 => pack::unpack_oct16(p[0]).extend(0.0).to_array(),
                1 => pack::unpack_oct8(p[0] as u16).extend(0.0).to_array(),
                2 => {
                    let (t, s) = pack::unpack_tangent(p[0]);
                    t.extend(s).to_array()
                }
                3 => pack::unpack_snorm_10_10_10_2(p[0]).to_array(),
                4 => pack::unpack_unorm_10_10_10_2(p[0]).to_array(),
                _ => pack::unpack4x16float([p[0], p[1]]).to_array(),
            };
            if mode == 5 {
                for (a, b) in g.iter().zip(&cpu) {
                    // Halves are exact in f32: the same bits, or both NaN.
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{p:?}: gpu {g:?}, cpu {cpu:?}");
                }
            } else {
                assert_close("decode", (mode, p[0]), &g, &cpu, 2e-6);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------

/// The value of `const NAME: TYPE = VALUE;` in `text`, as the text between `=` and `;`.
fn wgsl_const<'t>(text: &'t str, name: &str) -> &'t str {
    let start = format!("const {name}:");
    let line = text.lines().find(|l| l.starts_with(&start)).unwrap_or_else(|| panic!("no const {name}"));
    let (_, value) = line.split_once('=').unwrap_or_else(|| panic!("{line}"));
    value.trim().trim_end_matches(';').trim()
}

/// The numbers of a WGSL literal or constructor (`0u`, `-32.0`, `vec3<f32>(a, b, c)`).
fn wgsl_numbers(value: &str) -> Vec<f64> {
    let inner = value.split_once('(').map_or(value, |(_, rest)| rest.trim_end_matches(')'));
    inner
        .split(',')
        .map(|s| s.trim().trim_end_matches(['u', 'i', 'f']).parse::<f64>().unwrap_or_else(|e| panic!("{s}: {e}")))
        .collect()
}

#[test]
fn ac_wp05_2_tone_mapper_numbering_names_and_constants_match_the_cpu() {
    use tonemap::ToneMapper;
    let text = manifest::source("lib/color.wgsl").expect("lib/color.wgsl is embedded");
    // The WGSL numbering is the order of ToneMapper::ALL, the constant names are the stable names.
    for (index, mapper) in ToneMapper::ALL.into_iter().enumerate() {
        let name = format!("TONE_MAPPER_{}", mapper.name().replace('-', "_").to_uppercase());
        assert_eq!(wgsl_numbers(wgsl_const(text, &name)), [index as f64], "{name}");
    }
    assert_eq!(ToneMapper::ALL.len(), text.lines().filter(|l| l.starts_with("const TONE_MAPPER_")).count());
    // The default mapper is Khronos PBR Neutral on both sides, and post/tonemap.wgsl defaults to it.
    assert_eq!(ToneMapper::default(), ToneMapper::ALL[0]);
    assert_eq!(ToneMapper::from_name("pbr-neutral"), Some(ToneMapper::PbrNeutral));
    assert_eq!(ToneMapper::from_name("aces-fitted"), Some(ToneMapper::AcesFitted));
    let post = manifest::source("post/tonemap.wgsl").expect("post/tonemap.wgsl is embedded");
    assert!(post.lines().any(|l| l == "override TONE_MAPPER: u32 = 0u;"));

    let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    assert_eq!(as_f32(&wgsl_numbers(wgsl_const(text, "LUMINANCE_REC709"))), spaces::LUMINANCE_REC709_F32);
    assert_eq!(wgsl_numbers(wgsl_const(text, "EV100_MIN")), [tonemap::EV100_RANGE[0]]);
    assert_eq!(wgsl_numbers(wgsl_const(text, "EV100_MAX")), [tonemap::EV100_RANGE[1]]);
    for row in 0..3 {
        assert_eq!(
            wgsl_numbers(wgsl_const(text, &format!("ACES_FITTED_INPUT_{row}"))),
            tonemap::ACES_FITTED_INPUT[row]
        );
        assert_eq!(
            wgsl_numbers(wgsl_const(text, &format!("ACES_FITTED_OUTPUT_{row}"))),
            tonemap::ACES_FITTED_OUTPUT[row]
        );
    }
    // The PBR Neutral constants are private on the CPU side; their documented values.
    assert_eq!(wgsl_numbers(wgsl_const(text, "NEUTRAL_START_COMPRESSION")), [0.76]);
    assert_eq!(wgsl_numbers(wgsl_const(text, "NEUTRAL_DESATURATION")), [0.15]);
    let neutral_peak = tonemap::pbr_neutral([0.8, 0.8, 0.8]);
    assert!((neutral_peak[0] - 0.76).abs() < 1e-12, "the knee is where the CPU version starts too");
}

/// Linear HDR colours from deep shadow to far over white, plus greys and exact edges.
fn hdr_colours(rng: &mut Rng, n: usize) -> Vec<[f32; 3]> {
    let mut out =
        vec![[0.0; 3], [1.0; 3], [0.18; 3], [0.08; 3], [0.8; 3], [1e6, 1e6, 1e6], [1e30, 0.0, 0.5], [-0.5, 0.2, 2.0]];
    while out.len() < n {
        let grey = rng.chance(0.1);
        let c = [(); 3].map(|()| 10_f32.powf(rng.range_f32(-4.0, 3.0)));
        out.push(if grey { [c[0]; 3] } else { c });
    }
    out
}

#[test]
fn ac_wp05_2_tone_mappers_match_the_cpu() {
    use tonemap::ToneMapper;
    let mut rng = Rng::new(0x5EED_0503);
    let colours = hdr_colours(&mut rng, 20_000);
    let input: Vec<[u32; 4]> = colours.iter().map(|c| bits([c[0], c[1], c[2], 0.0])).collect();
    let n = input.len();
    let exposures =
        [1.0_f32, 0.37, tonemap::exposure_from_ev100(-2.0) as f32, tonemap::exposure_from_ev100(9.0) as f32];
    // Mode 7 is not a mapper: the shader clamps, like ToneMapper::Clamp.
    let modes = [(0, ToneMapper::ALL[0]), (1, ToneMapper::ALL[1]), (2, ToneMapper::ALL[2]), (7, ToneMapper::Clamp)];
    let mut worst = [0.0_f32; 3];
    for (mode, mapper) in modes {
        for exposure in exposures {
            let gpu = dispatch(&Run {
                count: n as u32,
                mode,
                a: exposure,
                input: &input,
                output_len: n,
                ..Run::new("test/color.wgsl", "tone_mappers")
            });
            for (c, g) in colours.iter().zip(&gpu) {
                let g = floats(*g);
                let cpu = mapper.apply_f32(*c, exposure);
                assert_close(mapper.name(), (c, exposure), &g[..3], &cpu, 1e-5);
                let error = g[..3].iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let slot = &mut worst[mode.min(2) as usize];
                *slot = slot.max(error);
            }
        }
    }
    println!("tone mappers, worst |gpu - cpu| (pbr-neutral, aces-fitted, clamp): {worst:?}");
}

#[test]
fn ac_wp05_2_srgb_curve_luminance_and_exposure_match_the_cpu() {
    let mut rng = Rng::new(0x5EED_0504);
    let mut values: Vec<[f32; 4]> = (0..20_000)
        .map(|_| {
            [rng.range_f32(-0.2, 1.2), rng.f32(), 10_f32.powf(rng.range_f32(-6.0, 0.5)), rng.range_f32(-40.0, 70.0)]
        })
        .collect();
    values.extend([
        [0.0, 0.003_130_8, 0.04045, 0.0],
        [1.0, -1.0, 0.5, -32.0],
        [f32::from_bits(0.003_130_8_f32.to_bits() + 1), 1e-9, 0.25, 64.0],
    ]);
    let input: Vec<[u32; 4]> = values.iter().copied().map(bits).collect();
    let n = input.len();
    for mode in [0, 1] {
        let gpu = dispatch(&Run {
            count: n as u32,
            mode,
            input: &input,
            output_len: n,
            ..Run::new("test/color.wgsl", "transfer")
        });
        for (v, g) in values.iter().zip(&gpu) {
            let g = floats(*g);
            let rgb = [v[0], v[1], v[2]];
            if mode == 0 {
                let cpu = transfer::linear_to_srgb_rgb_f32(rgb);
                assert_close("linear_to_srgb", v, &g[..3], &cpu, 2e-6 * scale(&cpu));
                assert_close("luminance", v, &g[3..], &[spaces::luminance_f32(rgb)], 1e-6);
            } else {
                let cpu = transfer::srgb_to_linear_rgb_f32(rgb);
                assert_close("srgb_to_linear", v, &g[..3], &cpu, 2e-6 * scale(&cpu));
                let exposure = tonemap::exposure_from_ev100(f64::from(v[3]));
                let relative = (f64::from(g[3]) - exposure).abs() / exposure;
                assert!(relative < 1e-6, "exposure_from_ev100({}): gpu {}, cpu {exposure}", v[3], g[3]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------------------------

#[test]
fn ac_wp05_2_quaternions_and_safe_normalize_match_the_cpu() {
    let mut rng = Rng::new(0x5EED_0505);
    let unit_quat = |rng: &mut Rng| {
        Quat::from_xyzw(rng.normal_f32(), rng.normal_f32(), rng.normal_f32(), rng.normal_f32()).normalize()
    };
    let mut cases: Vec<(Quat, Quat, Vec3)> = (0..10_000)
        .map(|_| {
            let v = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()) * rng.range_f32(0.01, 100.0);
            (unit_quat(&mut rng), unit_quat(&mut rng), v)
        })
        .collect();
    let q = unit_quat(&mut rng);
    cases.extend([(Quat::IDENTITY, q, Vec3::ZERO), (q, Quat::IDENTITY, Vec3::new(1e30, 1e30, 0.0))]);
    let input: Vec<[u32; 4]> = cases
        .iter()
        .flat_map(|(q, r, v)| [bits(q.to_array()), bits(r.to_array()), bits(v.extend(0.0).to_array())])
        .collect();
    let n = cases.len();
    let gpu = dispatch(&Run {
        count: n as u32,
        input: &input,
        output_len: 3 * n,
        ..Run::new("test/math.wgsl", "quaternions")
    });
    for (i, (q, r, v)) in cases.iter().enumerate() {
        let rotated = floats(gpu[3 * i]);
        let cpu = q.mul_vec3(*v);
        assert_close("quat_rotate", (q, v), &rotated[..3], &cpu.to_array(), 2e-6 * v.length().max(1.0));
        assert_close("quat_mul", (q, r), &floats(gpu[3 * i + 1]), &(*q * *r).to_array(), 2e-6);
        // safe_normalize: the WGSL rule (squared length in (1e-30, f32::MAX]) evaluated in f64.
        let d = DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z));
        let len2 = d.dot(d);
        let expected = if len2 > 1e-30 && len2 <= f64::from(f32::MAX) {
            let u = d / len2.sqrt();
            [u.x as f32, u.y as f32, u.z as f32]
        } else {
            [0.0, 0.0, 1.0]
        };
        assert_close("safe_normalize", v, &floats(gpu[3 * i + 2])[..3], &expected, 2e-6);
    }
}

#[test]
fn ac_wp05_2_orthonormal_basis_matches_the_cpu_and_is_a_right_handed_frame() {
    let mut rng = Rng::new(0x5EED_0506);
    let normals = directions(&mut rng, 20_000);
    // -0 in z is the one place the WGSL `n.z >= 0` and the CPU `copysign` choose differently.
    let normals: Vec<Vec3> = normals.into_iter().filter(|n| n.z.to_bits() != (-0.0_f32).to_bits()).collect();
    let input: Vec<[u32; 4]> = normals.iter().map(|n| bits(n.extend(0.0).to_array())).collect();
    let n = normals.len();
    let gpu =
        dispatch(&Run { count: n as u32, input: &input, output_len: 3 * n, ..Run::new("test/math.wgsl", "basis") });
    for (i, normal) in normals.iter().enumerate() {
        let column = |k: usize| {
            let c = floats(gpu[3 * i + k]);
            (Vec3::new(c[0], c[1], c[2]), c[3])
        };
        let ((t, _), (b, _), (nn, round_trip)) = (column(0), column(1), column(2));
        let d = DVec3::new(f64::from(normal.x), f64::from(normal.y), f64::from(normal.z));
        let (ct, cb) = d.any_orthonormal_pair();
        let narrow = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32];
        assert_close("tangent", normal, &t.to_array(), &narrow(ct), 2e-6);
        assert_close("bitangent", normal, &b.to_array(), &narrow(cb), 2e-6);
        assert_eq!(nn, *normal);
        assert!(
            (t.dot(b)).abs() < 1e-5 && (t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5,
            "{normal:?}"
        );
        assert!(t.cross(b).distance(*normal) < 1e-5, "right-handed: {normal:?}");
        assert!((round_trip - 1.0).abs() < 1e-5, "to_local(from_local(+Z)) = +Z: {normal:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// BRDF
// ---------------------------------------------------------------------------------------------

/// The theta rows of `furnace_rows` (test/brdf.wgsl THETA_STEPS) and its phi columns.
const THETA_STEPS: usize = 1024;
const PHI_STEPS: usize = 512;

/// GGX D in f64.
fn d_ggx(n_dot_h: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f64::consts::PI * d * d)
}

/// Smith G1 of GGX in f64.
fn g1_smith(n_dot_v: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    2.0 * n_dot_v / (n_dot_v + (a2 + (1.0 - a2) * n_dot_v * n_dot_v).sqrt())
}

/// The height-correlated Smith visibility of GGX in f64.
fn v_smith(n_dot_v: f64, n_dot_l: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let gl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (gv + gl).max(1e-20)
}

/// The single-scatter directional albedo of one GGX lobe with F = 1, by the midpoint rule on the
/// grid of `furnace_rows`, in f64.
fn albedo_cpu(n_dot_v: f64, alpha: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, TAU};
    let v = DVec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let (d_theta, d_phi) = (FRAC_PI_2 / THETA_STEPS as f64, TAU / PHI_STEPS as f64);
    let mut total = 0.0;
    for row in 0..THETA_STEPS {
        let theta = (row as f64 + 0.5) * d_theta;
        let mut sum = 0.0;
        for column in 0..PHI_STEPS {
            let phi = (column as f64 + 0.5) * d_phi;
            let l = DVec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            let h = (v + l).normalize();
            sum += d_ggx(h.z, alpha) * v_smith(n_dot_v, l.z, alpha) * l.z;
        }
        total += sum * theta.sin() * d_theta * d_phi;
    }
    total
}

/// The furnace integral of every cell `(n.v, alpha0, alpha1, blend)` for integrand `mode`.
fn furnace(cells: &[[f32; 4]], mode: u32) -> Vec<f64> {
    let input: Vec<[u32; 4]> = cells.iter().copied().map(bits).collect();
    let threads = cells.len() * THETA_STEPS;
    let rows = dispatch(&Run {
        count: threads as u32,
        mode,
        input: &input,
        output_len: threads,
        constants: &[("WORKGROUP_SIZE", 64.0), ("SAMPLE_COUNT", 1024.0)],
        ..Run::new("test/brdf.wgsl", "furnace_rows")
    });
    rows.as_chunks::<THETA_STEPS>()
        .0
        .iter()
        .map(|cell| cell.iter().map(|r| f64::from(f32::from_bits(r[0]))).sum())
        .collect()
}

/// The furnace grid: view cosines by GGX widths.
const VIEW_COSINES: [f32; 4] = [1.0, 0.7, 0.35, 0.1];
const ALPHAS: [f32; 4] = [0.1, 0.25, 0.5, 1.0];

/// One `(n.v, alpha)` cell per grid point.
fn grid_cells() -> Vec<[f32; 4]> {
    VIEW_COSINES.iter().flat_map(|&nv| ALPHAS.iter().map(move |&a| [nv, a, a, 0.0])).collect()
}

#[test]
fn ac_wp05_2_white_furnace() {
    let cells = grid_cells();
    // Lambert with unit albedo reflects exactly everything.
    for (cell, e) in cells.iter().zip(furnace(&cells, 3)) {
        assert!((e - 1.0).abs() < 1e-5, "Lambert {cell:?}: {e}");
    }
    // The projected GGX distribution integrates to one: integral of D(h) (n.h) over the hemisphere.
    for (cell, e) in cells.iter().zip(furnace(&cells, 2)) {
        assert!((e - 1.0).abs() < 2e-3, "GGX normalisation {cell:?}: {e}");
    }
    // The single-scatter specular albedo with F = 1 never exceeds one, falls with roughness, is
    // nearly one for a smooth lobe, and equals an f64 CPU quadrature of the same integrand.
    let specular = furnace(&cells, 0);
    let mut lines = Vec::new();
    for (i, (cell, &e)) in cells.iter().zip(&specular).enumerate() {
        assert!(e <= 1.0 + 1e-3, "energy is created at {cell:?}: {e}");
        let cpu = albedo_cpu(f64::from(cell[0]), f64::from(cell[1]));
        assert!((e - cpu).abs() < 2e-4, "{cell:?}: gpu {e}, cpu {cpu}");
        lines.push(format!("n.v {} alpha {}: {e:.4}", cell[0], cell[1]));
        if cell[1] == 1.0 {
            // At alpha = 1, D = 1/pi and V = 1 / (2 (n.l + n.v)): E = 1 - mu ln((1 + mu) / mu).
            let mu = f64::from(cell[0]);
            let exact = 1.0 - mu * ((1.0 + mu) / mu).ln();
            assert!((e - exact).abs() < 1e-4, "alpha 1, n.v {mu}: {e}, closed form {exact}");
        }
        if i % ALPHAS.len() > 0 && cell[0] >= 0.35 {
            // Away from grazing, a rougher lobe loses more (at grazing the visibility term makes it
            // non-monotonic: E(0.1, 0.25) > E(0.1, 0.1)).
            assert!(e < specular[i - 1], "albedo grows with roughness at {cell:?}");
        }
    }
    assert!(specular[0] > 0.98, "a smooth lobe at normal incidence keeps its energy: {}", specular[0]);
    println!("single-scatter GGX albedo E(n.v, alpha), F = 1:\n  {}", lines.join("\n  "));
    // The dual lobe is the blend of its two lobes (the integral is linear in the mix).
    let nv = 0.6;
    let (alpha0, alpha1) = (0.25, 0.6);
    let pair = furnace(&[[nv, alpha0, alpha0, 0.0], [nv, alpha1, alpha1, 0.0]], 0);
    let blends = [0.0, 0.15, 0.5, 1.0];
    let dual_cells: Vec<[f32; 4]> = blends.iter().map(|&b| [nv, alpha0, alpha1, b]).collect();
    for (blend, e) in blends.iter().zip(furnace(&dual_cells, 1)) {
        let expected = pair[0] + (pair[1] - pair[0]) * f64::from(*blend);
        assert!((e - expected).abs() < 2e-4, "dual lobe blend {blend}: {e}, expected {expected}");
    }
}

#[test]
fn ac_wp05_2_split_sum_dfg_matches_the_furnace_for_every_combination() {
    let cells = grid_cells();
    let albedo = furnace(&cells, 0);
    let input: Vec<[u32; 4]> = cells.iter().copied().map(bits).collect();
    let desc = manifest::shader("test/brdf.wgsl").expect("in the manifest");
    let mut values = [("", 0.0); manifest::MAX_OVERRIDES];
    // Results per SAMPLE_COUNT, which must not depend on WORKGROUP_SIZE.
    let mut by_samples: Vec<(f64, Vec<[u32; 4]>)> = Vec::new();
    for index in 0..desc.combination_count() {
        let constants = desc.combination(index, &mut values).expect("few overrides").to_vec();
        let samples = constants.iter().find(|(n, _)| *n == "SAMPLE_COUNT").map(|(_, v)| *v).expect("SAMPLE_COUNT");
        let gpu = dispatch(&Run {
            count: cells.len() as u32,
            input: &input,
            output_len: cells.len(),
            constants: &constants,
            ..Run::new("test/brdf.wgsl", "dfg")
        });
        let tolerance = if samples >= 1024.0 { 4e-3 } else { 3e-2 };
        for ((cell, g), e) in cells.iter().zip(&gpu).zip(&albedo) {
            let [a, b, sum, compensation] = floats(*g);
            assert!(a >= 0.0 && b >= 0.0 && sum <= 1.0 + 1e-3, "{cell:?}: A {a}, B {b}");
            assert!((f64::from(sum) - e).abs() < tolerance, "{constants:?} {cell:?}: A + B = {sum}, furnace {e}");
            let expected = 1.0 + 0.028 * (1.0 / sum.max(1e-4) - 1.0);
            assert!((compensation - expected).abs() < 1e-5 && compensation >= 1.0, "{cell:?}: {compensation}");
        }
        match by_samples.iter().find(|(s, _)| *s == samples) {
            Some((_, first)) => assert_eq!(first, &gpu, "{constants:?}: the workgroup size changed the result"),
            None => by_samples.push((samples, gpu)),
        }
    }
    assert_eq!(by_samples.len(), 3, "every SAMPLE_COUNT ran");
}

#[test]
fn ac_wp05_2_fresnel_and_masking_match_f64() {
    let mut rng = Rng::new(0x5EED_0507);
    let mut cases: Vec<[f32; 4]> = (0..10_000)
        .map(|_| [rng.range_f32(0.0, 1.0), rng.range_f32(0.5, 2.5), rng.range_f32(0.0, 1.0), rng.range_f32(0.01, 1.0)])
        .collect();
    cases.extend([[1.0, 1.4, 0.028, 1.0], [0.0, 1.4, 0.5, 0.01], [0.1, 0.7, 0.04, 0.5], [1.0, 1.0, 1.0, 1.0]]);
    let input: Vec<[u32; 4]> = cases.iter().copied().map(bits).collect();
    let n = cases.len();
    for alpha in [0.05_f32, 0.3, 1.0] {
        let gpu = dispatch(&Run {
            count: n as u32,
            a: alpha,
            input: &input,
            output_len: n,
            ..Run::new("test/brdf.wgsl", "fresnel")
        });
        for (c, g) in cases.iter().zip(&gpu) {
            let [cos_i, eta, f0, nv] = c.map(f64::from);
            let dielectric = {
                let g2 = eta * eta - 1.0 + cos_i * cos_i;
                if g2 < 0.0 {
                    1.0
                } else {
                    let g = g2.sqrt();
                    let a = (g - cos_i) / (g + cos_i);
                    let b = (cos_i * (g + cos_i) - 1.0) / (cos_i * (g - cos_i) + 1.0);
                    0.5 * a * a * (1.0 + b * b)
                }
            };
            let schlick = f0 + (1.0 - f0) * (1.0 - cos_i).clamp(0.0, 1.0).powi(5);
            let f0_ior = ((eta - 1.0) / (eta + 1.0)).powi(2);
            let cpu = [dielectric, schlick, f0_ior, g1_smith(nv, f64::from(alpha))].map(|x| x as f32);
            // WGSL allows 2.5 ULP for a division and the inherited accuracy of `sqrt`; the exact
            // Fresnel chains several of each, and a software adapter (lavapipe) misses 1e-5 by
            // 3 % on one case.
            let tolerance = if cos_i < 1e-3 { 1e-4 } else { 2e-5 };
            assert_close("fresnel", c, &floats(*g), &cpu, tolerance);
        }
    }
    // At normal incidence the exact Fresnel equals the f0 of the index; skin's f0 is 0.028.
    let skin = floats(
        dispatch(&Run {
            count: 1,
            a: 0.5,
            input: &[bits([1.0, 1.4, 0.0, 1.0])],
            output_len: 1,
            ..Run::new("test/brdf.wgsl", "fresnel")
        })[0],
    );
    assert!((skin[0] - skin[2]).abs() < 1e-6 && (skin[2] - 0.027_777_8).abs() < 1e-6, "{skin:?}");
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

/// The 0.32 fixed-point fraction `bits`, truncated to 24 bits, as in lib/sampling.wgsl.
fn fixed_to_unit(bits: u32) -> f32 {
    (bits >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[test]
fn ac_wp05_2_sequences_match_the_cpu_bit_for_bit() {
    let n = 1 << 14;
    let gpu = dispatch(&Run { count: n, output_len: n as usize, ..Run::new("test/sampling.wgsl", "sequences") });
    for (i, g) in (0..n).zip(&gpu) {
        let cpu = [
            i as f32 / n as f32,
            fixed_to_unit(i.reverse_bits()),
            fixed_to_unit(i.wrapping_mul(0xC13F_A9A9).wrapping_add(0x8000_0000)),
            fixed_to_unit(i.wrapping_mul(0x91E1_0DA6).wrapping_add(0x8000_0000)),
        ];
        assert_eq!(floats(*g), cpu, "point {i}");
        assert!(cpu.iter().all(|&x| (0.0..1.0).contains(&x)));
    }
    // R2 is fract(0.5 + i (1/g, 1/g^2)) with g the plastic number.
    let g = 1.324_717_957_244_746_f64;
    let i = 12_345_u32;
    let point = floats(gpu[i as usize]);
    let exact = [(0.5 + f64::from(i) / g).fract(), (0.5 + f64::from(i) / (g * g)).fract()];
    assert!(
        (f64::from(point[2]) - exact[0]).abs() < 1e-6 && (f64::from(point[3]) - exact[1]).abs() < 1e-6,
        "{point:?} {exact:?}"
    );
}

/// The sample means of `f` over `points`.
fn mean<const K: usize>(points: &[[f32; 4]], f: impl Fn(&[f64; 4]) -> [f64; K]) -> [f64; K] {
    let mut sum = [0.0; K];
    for p in points {
        let values = f(&p.map(f64::from));
        for (s, v) in sum.iter_mut().zip(values) {
            *s += v;
        }
    }
    sum.map(|s| s / points.len() as f64)
}

/// The integrals of `density` times (1, x, y, z, z^2) over the +Z hemisphere, by the midpoint rule.
fn hemisphere_moments(density: impl Fn(DVec3) -> f64) -> [f64; 5] {
    use std::f64::consts::{FRAC_PI_2, TAU};
    let (rows, columns) = (2048, 1024);
    let (d_theta, d_phi) = (FRAC_PI_2 / f64::from(rows), TAU / f64::from(columns));
    let mut out = [0.0; 5];
    for row in 0..rows {
        let theta = (f64::from(row) + 0.5) * d_theta;
        for column in 0..columns {
            let phi = (f64::from(column) + 0.5) * d_phi;
            let h = DVec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            let w = density(h) * theta.sin() * d_theta * d_phi;
            for (o, m) in out.iter_mut().zip([1.0, h.x, h.y, h.z, h.z * h.z]) {
                *o += w * m;
            }
        }
    }
    out
}

/// The warped Hammersley points of `mode` (see test/sampling.wgsl `warps`).
fn warps(mode: u32, alpha: f32, view_cos: f32) -> Vec<[f32; 4]> {
    let n = 1 << 16;
    let gpu = dispatch(&Run {
        count: n,
        mode,
        a: alpha,
        b: view_cos,
        output_len: n as usize,
        ..Run::new("test/sampling.wgsl", "warps")
    });
    gpu.into_iter().map(floats).collect()
}

#[track_caller]
fn assert_moments<const K: usize>(what: &str, got: [f64; K], expected: [f64; K], tolerance: f64) {
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < tolerance, "{what}: moments {got:?}, expected {expected:?}");
    }
}

#[test]
fn ac_wp05_2_warps_have_the_moments_of_their_densities() {
    use std::f64::consts::PI;
    // Uniform sphere: zero mean, E[x^2] = E[z^2] = 1/3, density 1 / (4 pi).
    let points = warps(0, 0.0, 0.0);
    for p in &points {
        assert!(
            (Vec3::new(p[0], p[1], p[2]).length() - 1.0).abs() < 1e-5 && (f64::from(p[3]) - 0.25 / PI).abs() < 1e-7
        );
    }
    assert_moments(
        "sphere",
        mean(&points, |p| [p[0], p[1], p[2], p[0] * p[0], p[2] * p[2]]),
        [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0],
        1e-3,
    );

    // Uniform hemisphere: E[z] = 1/2, E[z^2] = 1/3.
    let points = warps(1, 0.0, 0.0);
    assert!(points.iter().all(|p| p[2] >= 0.0));
    assert_moments("hemisphere", mean(&points, |p| [p[0], p[1], p[2], p[2] * p[2]]), [0.0, 0.0, 0.5, 1.0 / 3.0], 1e-3);

    // Cosine hemisphere: E[z] = 2/3, E[z^2] = 1/2, density cos / pi.
    let points = warps(2, 0.0, 0.0);
    for p in &points {
        assert!(p[2] >= 0.0 && (f64::from(p[3]) - f64::from(p[2]) / PI).abs() < 1e-6, "{p:?}");
    }
    assert_moments("cosine", mean(&points, |p| [p[0], p[1], p[2], p[2] * p[2]]), [0.0, 0.0, 2.0 / 3.0, 0.5], 1e-3);

    // Concentric disk: inside the unit disk, uniform by area: E[r^2] = 1/2, E[x^2] = 1/4.
    let points = warps(5, 0.0, 0.0);
    assert!(points.iter().all(|p| p[0] * p[0] + p[1] * p[1] <= 1.0 + 1e-6));
    assert_moments(
        "disk",
        mean(&points, |p| [p[0], p[1], p[0] * p[0] + p[1] * p[1], p[0] * p[0]]),
        [0.0, 0.0, 0.5, 0.25],
        1e-3,
    );

    for alpha in [0.2_f32, 0.5, 0.9] {
        let a = f64::from(alpha);
        // GGX NDF: density D(h) cos. E[1 / D] = integral of cos over the hemisphere = pi (1/D is
        // bounded, so the estimate converges), the density matches f64, and the moments match an
        // f64 quadrature of D cos.
        let points = warps(3, alpha, 0.0);
        for p in &points {
            let cpu = d_ggx(f64::from(p[2]), a) * f64::from(p[2]);
            assert!((f64::from(p[3]) - cpu).abs() <= 1e-4 * cpu.max(1.0), "NDF density at {p:?}: {cpu}");
        }
        let inverse_d = mean(&points, |p| [1.0 / d_ggx(p[2], a)]);
        assert!((inverse_d[0] - PI).abs() < 3e-3 * PI, "alpha {alpha}: E[1/D] = {}", inverse_d[0]);
        let quadrature = hemisphere_moments(|h| d_ggx(h.z, a) * h.z);
        assert!((quadrature[0] - 1.0).abs() < 1e-3, "D cos integrates to one: {}", quadrature[0]);
        let got = mean(&points, |p| [p[0], p[1], p[2], p[2] * p[2]]);
        assert_moments("GGX NDF", got, [quadrature[1], quadrature[2], quadrature[3], quadrature[4]], 2e-3);

        // GGX VNDF for three views: the density the library reports matches f64, integrates to one,
        // and the sample moments match its quadrature.
        for view_cos in [1.0_f32, 0.6, 0.15] {
            let points = warps(4, alpha, view_cos);
            let vc = f64::from(view_cos);
            let v = DVec3::new((1.0 - vc * vc).sqrt(), 0.0, vc);
            let pdf = |h: DVec3| g1_smith(v.z, a) * v.dot(h).max(0.0) * d_ggx(h.z, a) / v.z;
            for p in &points {
                let h = DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                assert!((h.length() - 1.0).abs() < 1e-5 && h.z >= 0.0 && v.dot(h) > -1e-5, "VNDF sample {p:?}");
                let cpu = pdf(h);
                assert!((f64::from(p[3]) - cpu).abs() <= 2e-4 * cpu.max(1.0), "VNDF density at {p:?}: {cpu}");
            }
            let quadrature = hemisphere_moments(pdf);
            assert!((quadrature[0] - 1.0).abs() < 2e-3, "the VNDF density integrates to one: {}", quadrature[0]);
            let got = mean(&points, |p| [p[0], p[1], p[2], p[2] * p[2]]);
            assert_moments(
                &format!("GGX VNDF alpha {alpha} n.v {view_cos}"),
                got,
                [quadrature[1], quadrature[2], quadrature[3], quadrature[4]],
                3e-3,
            );
        }
    }
}
