//! GPU tests of mip generation and the BC encoders, on the real adapter, headless.
//!
//! Evidence for `[SKN-04]` / `[AC-WP28-2]`: the CPU reference decoders agree bit for bit with the
//! GPU's hardware decoder on random blocks of every mode; encoder output decodes identically on the
//! CPU and in hardware; PSNR is measured per content and quality on fixed-seed synthetic images.
//! Evidence for `[IO-05]`: GPU mips match `humanart_io::mip` within one 8-bit step.

use std::fmt::Write as _;
use std::sync::{Mutex, MutexGuard, OnceLock};

use humanart_core::rng::Rng;
use humanart_gpu::profiler::GpuProfiler;
use humanart_gpu::resources::{Budget, Resources};
use humanart_gpu::{Capability, Gpu};
use humanart_io::bc::{self, BcFormat};
use humanart_io::{ColorSpace, Format, Image, Limits, Samples, ktx2, mip};
use humanart_math::{f16_bits_to_f32, f32_to_f16_bits};

use super::offline::{Request, halves_to_f32, read_buffer_blocking, to_ktx2};
use super::{Bc7Options, Compressor, Quality};

fn gpu() -> &'static Gpu {
    // The process's shared device (ADR-040).
    let gpu = humanart_gpu::shared_headless().expect("this machine has a GPU");
    assert!(gpu.has(Capability::TextureCompressionBc), "the test adapter must decode BC textures");
    gpu
}

fn compressor() -> MutexGuard<'static, Compressor> {
    static COMPRESSOR: OnceLock<Mutex<Compressor>> = OnceLock::new();
    COMPRESSOR
        .get_or_init(|| Mutex::new(Compressor::new(gpu()).expect("every texture pipeline builds")))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Uploads a block stream as a BC texture and reads every texel back through `textureLoad`, i.e.
/// through the GPU's own decoder.
fn hardware_decode(format: wgpu::TextureFormat, blocks: &[u8], width: u32, height: u32) -> Vec<[f32; 4]> {
    let gpu = gpu();
    assert!(width.is_multiple_of(4) && height.is_multiple_of(4), "BC textures are whole blocks");
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bc under test"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        blocks,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width / 4 * 16),
            rows_per_image: Some(height / 4),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    let source = humanart_shaders::compose("texture/bc_fetch.wgsl", &[]).expect("composes").into_source();
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bc fetch"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bc fetch"),
        layout: None,
        module: &module,
        entry_point: Some("fetch"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let size = u64::from(width) * u64::from(height) * 16;
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fetched texels"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bc fetch"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fetch") });
    {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fetch"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes = read_buffer_blocking(gpu, &out, size).expect("readback");
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let f = |i: usize| f32::from_le_bytes([t[4 * i], t[4 * i + 1], t[4 * i + 2], t[4 * i + 3]]);
            [f(0), f(1), f(2), f(3)]
        })
        .collect()
}

/// Hardware-decoded BC7 texels as 8-bit values.
fn hardware_rgba8(blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    hardware_decode(wgpu::TextureFormat::Bc7RgbaUnorm, blocks, width, height)
        .iter()
        .flat_map(|t| t.map(|v| (v * 255.0).round() as u8))
        .collect()
}

/// CPU-decoded BC7 texels.
fn cpu_rgba8(blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    let image =
        bc::decode(BcFormat::Bc7, blocks, width, height, ColorSpace::Linear, &Limits::DEFAULT).expect("decodes");
    image.as_u8().expect("8-bit").to_vec()
}

/// Random blocks with the mode field forced to cycle through every mode (and the reserved ones).
fn random_blocks(count: usize, seed: u64, codes: &[(u8, u8)]) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut blocks = vec![0_u8; count * 16];
    rng.fill(&mut blocks);
    for (i, block) in blocks.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let (code, bits) = codes[i % codes.len()];
        let mask = ((1_u16 << bits) - 1) as u8;
        block[0] = (block[0] & !mask) | code;
    }
    blocks
}

#[test]
fn skn_04_hardware_bc7_decoder_matches_the_cpu_reference_on_random_blocks() {
    // Mode m is m zero bits then a one; code 0 with 8 bits is the reserved mode 8.
    let codes: Vec<(u8, u8)> = (0..8).map(|m| (1_u8 << m, m + 1)).chain([(0, 8)]).collect();
    let (w, h) = (256, 256);
    let blocks = random_blocks(64 * 64, 0xBC7, &codes);
    let gpu = hardware_rgba8(&blocks, w, h);
    let cpu = cpu_rgba8(&blocks, w, h);
    let mut bad = Vec::new();
    for (b, chunk) in blocks.as_chunks::<16>().0.iter().enumerate() {
        let (bx, by) = (b % 64, b / 64);
        for t in 0..16 {
            let at = ((by * 4 + t / 4) * w as usize + bx * 4 + t % 4) * 4;
            if gpu[at..at + 4] != cpu[at..at + 4] && bad.len() < 8 {
                bad.push((bc::bc7::block_mode(chunk), b, t, &gpu[at..at + 4], &cpu[at..at + 4]));
            }
        }
    }
    assert!(bad.is_empty(), "hardware vs CPU (mode, block, texel, gpu, cpu): {bad:?}");
}

#[test]
fn skn_04_hardware_bc6h_decoder_matches_the_cpu_reference_on_random_blocks() {
    let mut codes: Vec<(u8, u8)> = bc::bc6h::MODES.iter().map(|m| (m.code, m.code_bits)).collect();
    codes.extend([(0b10011, 5), (0b11111, 5)]);
    let (w, h) = (256, 256);
    for (signed, format) in [(false, wgpu::TextureFormat::Bc6hRgbUfloat), (true, wgpu::TextureFormat::Bc6hRgbFloat)] {
        let blocks = random_blocks(64 * 64, 0xBC6 + u64::from(signed), &codes);
        let gpu = hardware_decode(format, &blocks, w, h);
        let fmt = if signed { BcFormat::Bc6hSfloat } else { BcFormat::Bc6hUfloat };
        let cpu = bc::decode(fmt, &blocks, w, h, ColorSpace::Linear, &Limits::DEFAULT).expect("decodes");
        let cpu = match &cpu.samples {
            Samples::F16(v) => v.clone(),
            _ => panic!("half floats expected"),
        };
        let mut bad = Vec::new();
        let mut per_mode = [0_u32; 14];
        for (b, chunk) in blocks.as_chunks::<16>().0.iter().enumerate() {
            let mode = bc::bc6h::block_mode(chunk);
            let (bx, by) = (b % 64, b / 64);
            for t in 0..16 {
                let p = (by * 4 + t / 4) * w as usize + bx * 4 + t % 4;
                let g = [0, 1, 2].map(|c| f32_to_f16_bits(gpu[p][c]));
                let c = [cpu[3 * p], cpu[3 * p + 1], cpu[3 * p + 2]];
                // -0 and +0 are one value: a decoder may give either for a zero endpoint
                // (llvmpipe gives -0 where the reference gives +0 in signed mode).
                let zero = |h: u16| if h == 0x8000 { 0 } else { h };
                if g.map(zero) != c.map(zero) {
                    if let Some(m) = mode {
                        per_mode[m] += 1;
                    }
                    if bad.len() < 6 {
                        bad.push((mode, b, t, g, c));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "signed {signed}: mismatching texels per mode {per_mode:?}; (mode, block, texel, gpu, cpu): {bad:?}"
        );
    }
}

// ---- synthetic content -------------------------------------------------------------------------

/// Smooth value noise on a wrapping lattice of `n × n` random values.
struct Noise {
    n: usize,
    lattice: Vec<f32>,
}

impl Noise {
    fn new(seed: u64, n: usize) -> Self {
        let mut rng = Rng::new(seed);
        Self { n, lattice: (0..n * n).map(|_| rng.f32()).collect() }
    }

    fn at(&self, x: f32, y: f32) -> f32 {
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = (x - fx, y - fy);
        let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let n = self.n as i64;
        let v = |i: i64, j: i64| self.lattice[(j.rem_euclid(n) * n + i.rem_euclid(n)) as usize];
        let (i, j) = (fx as i64, fy as i64);
        let a = v(i, j) + (v(i + 1, j) - v(i, j)) * sx;
        let b = v(i, j + 1) + (v(i + 1, j + 1) - v(i, j + 1)) * sx;
        a + (b - a) * sy
    }

    /// Fractal sum of `octaves` octaves starting at `freq` lattice cells per image, in 0..1.
    fn fbm(&self, u: f32, v: f32, freq: f32, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut f, mut norm) = (0.0, 1.0, freq, 0.0);
        for _ in 0..octaves {
            sum += amp * self.at(u * f, v * f);
            norm += amp;
            amp *= 0.5;
            f *= 2.0;
        }
        sum / norm
    }
}

fn rgba8(w: u32, h: u32, space: ColorSpace, f: impl Fn(f32, f32, u32, u32) -> [u8; 4]) -> Image {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend(f(x as f32 / w as f32, y as f32 / h as f32, x, y));
        }
    }
    Image::new(Format::Bc, w, h, 4, space, Samples::U8(px)).expect("valid")
}

fn to8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Skin-like albedo: a melanin / haemoglobin field of low-frequency noise around a skin tone, fine
/// pores and a few darker freckles; sRGB-encoded.
fn skin(w: u32, h: u32) -> Image {
    let low = Noise::new(11, 16);
    let high = Noise::new(12, 64);
    rgba8(w, h, ColorSpace::Srgb, |u, v, _, _| {
        let melanin = low.fbm(u, v, 3.0, 4);
        let blood = low.fbm(v + 0.37, u + 0.11, 5.0, 3);
        let pores = high.fbm(u, v, 48.0, 2);
        let pore = if pores < 0.32 { 0.82 } else { 1.0 };
        let freckle = if high.fbm(u + 0.5, v, 12.0, 2) > 0.74 { 0.8 } else { 1.0 };
        let base = [0.62 - 0.18 * melanin + 0.05 * blood, 0.42 - 0.16 * melanin - 0.04 * blood, 0.33 - 0.12 * melanin];
        let rgb = base.map(|c| humanart_math::color::transfer::linear_to_srgb_f32(c * c * pore * freckle * 1.6));
        [to8(rgb[0]), to8(rgb[1]), to8(rgb[2]), 255]
    })
}

/// Smooth gradients with a little noise.
fn gradients(w: u32, h: u32) -> Image {
    let mut rng = Rng::new(21);
    let noise: Vec<f32> = (0..w * h).map(|_| rng.range_f32(-3.0, 3.0)).collect();
    rgba8(w, h, ColorSpace::Srgb, |u, v, x, y| {
        let n = noise[(y * w + x) as usize] / 255.0;
        [to8(u + n), to8(v + n), to8(0.5 + 0.5 * (6.0 * u * v).sin() + n), 255]
    })
}

/// Shapes with hard and soft alpha edges.
fn alpha_shapes(w: u32, h: u32) -> Image {
    let n = Noise::new(31, 16);
    rgba8(w, h, ColorSpace::Srgb, |u, v, _, _| {
        let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
        let a = if d < 0.3 {
            1.0
        } else if d < 0.4 {
            (0.4 - d) * 10.0
        } else {
            0.0
        };
        let hard = if (u * 8.0).fract() < 0.5 { 1.0 } else { 0.0 };
        [to8(n.fbm(u, v, 4.0, 3)), to8(u), to8(hard), to8(a.max(hard * 0.5))]
    })
}

/// Uniform random texels: the worst case.
fn random_image(w: u32, h: u32) -> Image {
    let mut rng = Rng::new(41);
    let mut px = vec![0_u8; (w * h * 4) as usize];
    rng.fill(&mut px);
    for p in px.as_chunks_mut::<4>().0.iter_mut() {
        p[3] = 255;
    }
    Image::new(Format::Bc, w, h, 4, ColorSpace::Srgb, Samples::U8(px)).expect("valid")
}

/// A tangent-space normal map from a height field (central differences), roughness in alpha.
fn normal_map(w: u32, h: u32) -> Image {
    let n = Noise::new(51, 32);
    let height = |x: f32, y: f32| n.fbm(x, y, 6.0, 5);
    let (du, dv) = (1.0 / w as f32, 1.0 / h as f32);
    rgba8(w, h, ColorSpace::Normal, |u, v, _, _| {
        let sx = (height(u + du, v) - height(u - du, v)) * 0.5 * w as f32 * 0.04;
        let sy = (height(u, v + dv) - height(u, v - dv)) * 0.5 * h as f32 * 0.04;
        let len = (sx * sx + sy * sy + 1.0).sqrt();
        let nn = [-sx / len, -sy / len, 1.0 / len];
        let rough = 0.35 + 0.3 * n.fbm(v, u, 3.0, 2);
        [to8(nn[0] * 0.5 + 0.5), to8(nn[1] * 0.5 + 0.5), to8(nn[2] * 0.5 + 0.5), to8(rough)]
    })
}

fn rgb_f32(w: u32, h: u32, f: impl Fn(f32, f32) -> [f32; 3]) -> Image {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend(f(x as f32 / w as f32, y as f32 / h as f32));
        }
    }
    Image::new(Format::Bc, w, h, 3, ColorSpace::Linear, Samples::F32(px)).expect("valid")
}

/// HDR sky: a horizon gradient over four decades with clouds and a bright sun disk.
fn hdr_sky(w: u32, h: u32) -> Image {
    let n = Noise::new(61, 16);
    rgb_f32(w, h, |u, v| {
        let sky = 10.0_f32.powf(2.0 - 3.5 * v);
        let cloud = n.fbm(u, v, 4.0, 4);
        let d = ((u - 0.7).powi(2) + (v - 0.2).powi(2)).sqrt();
        let sun = if d < 0.03 { 2000.0 } else { 0.0 };
        [sky * (0.4 + cloud) + sun, sky * (0.6 + 0.5 * cloud) + sun, sky * (1.0 + 0.2 * cloud) + sun * 0.9]
    })
}

/// Signed data: smooth fields in -8..8.
fn hdr_signed(w: u32, h: u32) -> Image {
    let n = Noise::new(71, 16);
    rgb_f32(w, h, |u, v| [(n.fbm(u, v, 3.0, 4) - 0.5) * 16.0, (n.fbm(v, u, 5.0, 3) - 0.5) * 4.0, (u - 0.5) * 0.02])
}

// ---- encoder quality ---------------------------------------------------------------------------

/// Encodes level 0, decodes on the CPU and in hardware, checks they agree, returns the PSNR of the
/// selected channels.
fn bc7_round_trip(image: &Image, quality: Quality, weights: [f32; 4], mask: u32) -> f64 {
    let gpu = gpu();
    let request = Request { mips: false, bc7: Bc7Options { quality, weights }, ..Request::new(BcFormat::Bc7) };
    let encoded = compressor().compress_blocking(gpu, image, &request).expect("encodes");
    let (w, h) = (image.width, image.height);
    let cpu = cpu_rgba8(&encoded.blocks[0], w, h);
    let hw = hardware_rgba8(&encoded.blocks[0], w, h);
    assert!(cpu == hw, "the GPU decodes encoder output exactly like the CPU reference");
    bc::psnr(image.as_u8().expect("8-bit"), &cpu, 4, mask).expect("same size")
}

/// A BC7 quality case: name, image, channel weights, measured channels, PSNR floors per quality.
type Bc7Case = (&'static str, Image, [f32; 4], u32, [f64; 3]);

#[test]
fn skn_04_bc7_encoder_quality_table() {
    let (w, h) = (256, 256);
    let cases: [Bc7Case; 5] = [
        ("skin", skin(w, h), [1.0; 4], 0xF, [53.5, 55.5, 57.0]),
        ("gradients", gradients(w, h), [1.0; 4], 0xF, [47.5, 49.5, 50.0]),
        ("alpha shapes", alpha_shapes(w, h), [1.0; 4], 0xF, [51.0, 51.0, 52.0]),
        ("normal + roughness", normal_map(w, h), [1.0; 4], 0xF, [40.5, 41.5, 45.5]),
        ("random", random_image(w, h), [1.0, 1.0, 1.0, 0.0], 0x7, [13.0, 15.5, 17.0]),
    ];
    let mut table = String::from("BC7 PSNR (dB), 256x256, level 0: content | fast | normal | high\n");
    for (name, image, weights, mask, floors) in &cases {
        let mut row = format!("{name:>20}");
        for (q, floor) in Quality::ALL.into_iter().zip(floors) {
            let p = bc7_round_trip(image, q, *weights, *mask);
            let _ = write!(row, " | {p:6.2}");
            assert!(p >= *floor, "{name} at {q:?}: {p:.2} dB below {floor}");
        }
        table.push_str(&row);
        table.push('\n');
    }
    println!("{table}");
}

/// Constant blocks are exact whatever the parity of their channels (the mode 5 single-colour path
/// with `humanart_io::bc::tables::BC7_MODE5_SINGLE`).
#[test]
fn skn_04_bc7_constant_and_edge_blocks() {
    let exact = rgba8(8, 4, ColorSpace::Linear, |_, _, _, _| [17, 201, 33, 255]);
    let mixed = rgba8(12, 8, ColorSpace::Linear, |_, _, x, _| if x < 8 { [17, 200, 33, 255] } else { [1, 2, 254, 77] });
    for q in Quality::ALL {
        assert_eq!(bc7_round_trip(&exact, q, [1.0; 4], 0xF), f64::INFINITY, "{q:?}");
        let request =
            Request { mips: false, bc7: Bc7Options { quality: q, weights: [1.0; 4] }, ..Request::new(BcFormat::Bc7) };
        let encoded = compressor().compress_blocking(gpu(), &mixed, &request).expect("encodes");
        let decoded = cpu_rgba8(&encoded.blocks[0], 12, 8);
        assert_eq!(max_diff(mixed.as_u8().expect("8-bit"), &decoded), 0, "{q:?}");
    }
    // A size that is not a multiple of 4 is padded by edge replication and cropped on decode.
    let odd = skin(13, 7);
    let encoded = compressor()
        .compress_blocking(gpu(), &odd, &Request { mips: false, ..Request::new(BcFormat::Bc7) })
        .expect("encodes");
    assert_eq!(encoded.blocks[0].len(), 4 * 2 * 16);
    let cpu =
        bc::decode(BcFormat::Bc7, &encoded.blocks[0], 13, 7, ColorSpace::Srgb, &Limits::DEFAULT).expect("decodes");
    let p = bc::psnr(odd.as_u8().expect("8-bit"), cpu.as_u8().expect("8-bit"), 4, 0xF).expect("same size");
    assert!(p > 40.0, "{p}");
}

/// Every 8-bit value in every channel, with the other channels random, plus random colours and
/// greys with mixed parities: each constant block decodes to exactly its colour, at every quality.
#[test]
fn skn_04_bc7_every_constant_colour_round_trips_exactly() {
    let mut rng = Rng::new(0xBC7_5111);
    let mut colours: Vec<[u8; 4]> = Vec::with_capacity(4096);
    let byte = |rng: &mut Rng| (rng.next_u32() & 0xFF) as u8;
    for channel in 0..4 {
        for v in 0..=255_u8 {
            let mut c = [byte(&mut rng), byte(&mut rng), byte(&mut rng), 255];
            if channel < 3 && rng.below(2) == 0 {
                c[3] = byte(&mut rng);
            }
            c[channel] = v;
            colours.push(c);
        }
    }
    for v in 0..=255_u8 {
        colours.push([v, v ^ 1, 255 - v, 255]);
    }
    while colours.len() < 4096 {
        let c = [byte(&mut rng), byte(&mut rng), byte(&mut rng), byte(&mut rng)];
        colours.push(c);
    }
    let (bw, bh) = (64_u32, (colours.len() as u32).div_ceil(64));
    let image = rgba8(bw * 4, bh * 4, ColorSpace::Linear, |_, _, x, y| {
        colours.get(((y / 4) * bw + x / 4) as usize).copied().unwrap_or([0, 0, 0, 255])
    });
    for q in Quality::ALL {
        let request =
            Request { mips: false, bc7: Bc7Options { quality: q, weights: [1.0; 4] }, ..Request::new(BcFormat::Bc7) };
        let encoded = compressor().compress_blocking(gpu(), &image, &request).expect("encodes");
        let decoded = cpu_rgba8(&encoded.blocks[0], bw * 4, bh * 4);
        let reference = image.as_u8().expect("8-bit");
        let wrong = reference.as_chunks::<4>().0.iter().zip(decoded.as_chunks::<4>().0).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{q:?}: {wrong} texels of {} constant blocks differ", colours.len());
    }
}

/// Mean and maximum angle in degrees between the normals of two RGBA normal maps; with `xy` the
/// decoded z is ignored and reconstructed from x and y.
fn angular_error(reference: &[u8], decoded: &[u8], xy: bool) -> (f64, f64) {
    let unpack = |p: &[u8; 4], rebuild: bool| {
        let x = f64::from(p[0]) / 255.0 * 2.0 - 1.0;
        let y = f64::from(p[1]) / 255.0 * 2.0 - 1.0;
        let z = if rebuild { (1.0 - x * x - y * y).max(0.0).sqrt() } else { f64::from(p[2]) / 255.0 * 2.0 - 1.0 };
        let l = (x * x + y * y + z * z).sqrt().max(1e-12);
        [x / l, y / l, z / l]
    };
    let (mut sum, mut max, mut n) = (0.0, 0.0_f64, 0.0);
    for (a, b) in reference.as_chunks::<4>().0.iter().zip(decoded.as_chunks::<4>().0) {
        let (u, v) = (unpack(a, false), unpack(b, xy));
        let angle = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0).acos().to_degrees();
        sum += angle;
        max = max.max(angle);
        n += 1.0;
    }
    (sum / n, max)
}

/// Normal maps two ways: xyz stored with equal weights, and x, y weighted with z reconstructed in
/// the shader (`Bc7Options::normal_xy`). Prints the angular error table.
#[test]
fn skn_04_bc7_normal_maps_with_z_reconstruction() {
    let image = normal_map(256, 256);
    let reference = image.as_u8().expect("8-bit");
    let mut table =
        String::from("BC7 normal map angular error (deg, mean / max), 256x256: quality | xyz | xy + rebuilt z\n");
    for q in Quality::ALL {
        let mut row = format!("{q:>8?}");
        for (options, xy) in [(Bc7Options { quality: q, weights: [1.0; 4] }, false), (Bc7Options::normal_xy(q), true)] {
            let request = Request { mips: false, bc7: options, ..Request::new(BcFormat::Bc7) };
            let encoded = compressor().compress_blocking(gpu(), &image, &request).expect("encodes");
            let decoded = cpu_rgba8(&encoded.blocks[0], 256, 256);
            let (mean, max) = angular_error(reference, &decoded, xy);
            let _ = write!(row, " | {mean:.3} / {max:.2}");
            // Regression bounds a little above the values measured on the development adapter.
            let (mean_bound, max_bound) = match q {
                Quality::Fast => (1.7, 14.0),
                Quality::Normal => (1.5, 8.0),
                Quality::High => (1.0, 5.0),
            };
            assert!(mean < mean_bound && max < max_bound, "{q:?} xy {xy}: {mean} / {max}");
        }
        table.push_str(&row);
        table.push('\n');
    }
    println!("{table}");
}

fn bc6h_round_trip(image: &Image, signed: bool, quality: Quality) -> f64 {
    let gpu = gpu();
    let format = if signed { BcFormat::Bc6hSfloat } else { BcFormat::Bc6hUfloat };
    let mut request = Request { mips: false, ..Request::new(format) };
    request.bc6h.quality = quality;
    let encoded = compressor().compress_blocking(gpu, image, &request).expect("encodes");
    let (w, h) = (image.width, image.height);
    let cpu = bc::decode(format, &encoded.blocks[0], w, h, ColorSpace::Linear, &Limits::DEFAULT).expect("decodes");
    let cpu = match &cpu.samples {
        Samples::F16(v) => v.clone(),
        _ => panic!("half floats expected"),
    };
    let wf = if signed { wgpu::TextureFormat::Bc6hRgbFloat } else { wgpu::TextureFormat::Bc6hRgbUfloat };
    let hw = hardware_decode(wf, &encoded.blocks[0], w, h);
    let hw: Vec<u16> = hw.iter().flat_map(|t| [0, 1, 2].map(|c| f32_to_f16_bits(t[c]))).collect();
    assert!(cpu == hw, "the GPU decodes encoder output exactly like the CPU reference");
    // The reference is the input as the encoder saw it: rounded to half floats.
    let reference: Vec<f32> =
        image.as_f32().expect("f32").iter().map(|&v| f16_bits_to_f32(f32_to_f16_bits(v))).collect();
    bc::psnr_f16(&reference, &halves_to_f32(&cpu)).expect("same size")
}

#[test]
fn skn_04_bc6h_encoder_quality_table() {
    let (w, h) = (256, 256);
    let cases: [(&str, Image, bool, [f64; 3]); 3] = [
        ("hdr sky (UF16)", hdr_sky(w, h), false, [67.5, 68.0, 68.0]),
        ("signed fields (SF16)", hdr_signed(w, h), true, [38.5, 40.5, 40.5]),
        (
            "skin as float (UF16)",
            {
                let s = skin(w, h);
                let px: Vec<f32> = s
                    .as_u8()
                    .expect("8-bit")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [0, 1, 2].map(|c| humanart_math::color::transfer::srgb8_to_linear(p[c])))
                    .collect();
                Image::new(Format::Bc, w, h, 3, ColorSpace::Linear, Samples::F32(px)).expect("valid")
            },
            false,
            [52.5, 53.0, 53.0],
        ),
    ];
    let mut table = String::from("BC6H PSNR (dB, log2(1+x) domain), 256x256: content | fast | normal | high\n");
    for (name, image, signed, floors) in &cases {
        let mut row = format!("{name:>22}");
        for (q, floor) in Quality::ALL.into_iter().zip(floors) {
            let p = bc6h_round_trip(image, *signed, q);
            let _ = write!(row, " | {p:6.2}");
            assert!(p >= *floor, "{name} at {q:?}: {p:.2} dB below {floor}");
        }
        table.push_str(&row);
        table.push('\n');
    }
    println!("{table}");
}

// ---- mips --------------------------------------------------------------------------------------

/// Largest per-sample difference of two 8-bit arrays.
fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
}

#[test]
fn io_05_gpu_mips_match_the_cpu_reference() {
    let gpu = gpu();
    for (name, image) in [
        ("srgb skin 37x23", skin(37, 23)),
        ("srgb alpha 64x40", alpha_shapes(64, 40)),
        ("data 33x65", {
            let mut d = gradients(33, 65);
            d.color_space = ColorSpace::Data;
            d
        }),
        ("linear 16x16", {
            let mut d = gradients(16, 16);
            d.color_space = ColorSpace::Linear;
            d
        }),
    ] {
        let gpu_mips = compressor().mips_blocking(gpu, &image).expect("mips");
        let cpu_mips = mip::generate(&image, 0).expect("cpu mips");
        assert_eq!(gpu_mips.len(), cpu_mips.len(), "{name}");
        for (level, (g, c)) in gpu_mips.iter().zip(&cpu_mips).enumerate() {
            assert_eq!((g.width, g.height), (c.width, c.height), "{name} level {level}");
            let d = max_diff(g.as_u8().expect("8-bit"), c.as_u8().expect("8-bit"));
            assert!(d <= 1, "{name} level {level}: GPU and CPU differ by {d}");
        }
    }
}

#[test]
fn io_05_gpu_normal_mips_match_the_cpu_reference() {
    let gpu = gpu();
    let image = normal_map(48, 36);
    let gpu_mips = compressor().mips_blocking(gpu, &image).expect("mips");
    let px = image.as_u8().expect("8-bit");
    let normals = Image::new(
        Format::Bc,
        image.width,
        image.height,
        3,
        ColorSpace::Normal,
        Samples::U8(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()),
    )
    .expect("valid");
    let rough = Image::new(
        Format::Bc,
        image.width,
        image.height,
        1,
        ColorSpace::Data,
        Samples::U8(px.as_chunks::<4>().0.iter().map(|p| p[3]).collect()),
    )
    .expect("valid");
    let cpu = mip::generate_normal(&normals, Some(&rough), 0.0, 0).expect("cpu mips");
    let mut widened = 0;
    for (level, g) in gpu_mips.iter().enumerate() {
        let g = g.as_u8().expect("8-bit");
        let n = cpu.normals[level].as_u8().expect("8-bit");
        let r = cpu.roughness[level].as_u8().expect("8-bit");
        let gn: Vec<u8> = g.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let gr: Vec<u8> = g.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
        assert!(max_diff(&gn, n) <= 1, "level {level} normals");
        assert!(max_diff(&gr, r) <= 1, "level {level} roughness");
        if level > 0 {
            let mean_in: f32 =
                rough.as_u8().expect("8-bit").iter().map(|&v| f32::from(v)).sum::<f32>() / rough.pixel_count() as f32;
            let mean_out: f32 = gr.iter().map(|&v| f32::from(v)).sum::<f32>() / gr.len() as f32;
            if mean_out > mean_in + 1.0 {
                widened += 1;
            }
        }
    }
    assert!(widened >= 3, "the bumpy map gets rougher in the small mips ({widened} levels)");
}

#[test]
fn io_05_float_mips_match_the_cpu_reference() {
    let gpu = gpu();
    let image = hdr_sky(40, 24);
    let gpu_mips = compressor().mips_blocking(gpu, &image).expect("mips");
    let cpu_mips = mip::generate(&image, 0).expect("cpu mips");
    for (level, (g, c)) in gpu_mips.iter().zip(&cpu_mips).enumerate().skip(1) {
        let Samples::F16(gv) = &g.samples else { panic!("halves") };
        let cv = c.as_f32().expect("f32");
        for (p, cpx) in cv.as_chunks::<3>().0.iter().enumerate() {
            for ch in 0..3 {
                let gpu_v = f16_bits_to_f32(gv[4 * p + ch]);
                let rel = (gpu_v - cpx[ch]).abs() / cpx[ch].abs().max(1e-3);
                assert!(rel < 2e-3, "level {level} texel {p} channel {ch}: {gpu_v} vs {}", cpx[ch]);
            }
        }
    }
}

// ---- containers and pipelines ------------------------------------------------------------------

#[test]
fn skn_04_bc7_mip_chain_round_trips_through_ktx2() {
    let gpu = gpu();
    let image = skin(64, 48);
    let encoded = compressor().compress_blocking(gpu, &image, &Request::new(BcFormat::Bc7)).expect("encodes");
    assert_eq!(encoded.blocks.len(), 7);
    let texture = to_ktx2(&encoded).expect("ktx2 texture");
    let bytes = ktx2::write(&texture).expect("writes");
    let back = ktx2::read(&bytes, &Limits::DEFAULT).expect("reads");
    assert_eq!(back.format.vk_format, ktx2::vk::BC7_SRGB);
    assert_eq!((back.width, back.height), (64, 48));
    assert_eq!(back.levels, encoded.blocks);
    // Every level decodes to something close to the GPU-generated mip of the same level.
    let mips = compressor().mips_blocking(gpu, &image).expect("mips");
    for (level, (blocks, reference)) in back.levels.iter().zip(&mips).enumerate() {
        let decoded =
            bc::decode(BcFormat::Bc7, blocks, reference.width, reference.height, ColorSpace::Srgb, &Limits::DEFAULT)
                .expect("decodes");
        let p = bc::psnr(reference.as_u8().expect("8-bit"), decoded.as_u8().expect("8-bit"), 4, 0xF).expect("same");
        assert!(p > 38.0, "level {level}: {p:.2} dB");
    }
}

#[test]
fn perf_r4_every_texture_pipeline_is_prewarmed() {
    let gpu = gpu();
    let mut c = compressor();
    let image = gradients(32, 32);
    c.compress_blocking(gpu, &image, &Request::new(BcFormat::Bc7)).expect("encodes");
    let hdr = hdr_signed(16, 16);
    c.compress_blocking(gpu, &hdr, &Request::new(BcFormat::Bc6hSfloat)).expect("encodes");
    assert_eq!(c.late_pipelines(), 0);
}

#[test]
fn requests_are_checked() {
    let gpu = gpu();
    let mut c = compressor();
    let hdr = hdr_sky(8, 8);
    assert!(c.compress_blocking(gpu, &hdr, &Request::new(BcFormat::Bc7)).is_err(), "BC7 takes 8-bit images");
    let mut bad = Request::new(BcFormat::Bc7);
    bad.bc7.weights = [f32::NAN, 1.0, 1.0, 1.0];
    assert!(c.compress_blocking(gpu, &skin(8, 8), &bad).is_err());
}

/// `[AC-WP28-2]` throughput: 4096 × 4096 on this PC's GPU, encode only (the source is uploaded and
/// the pipelines are warm), averaged over 3 runs after a warm-up. Run with
/// `cargo test -p humanart-render --release -- --ignored --nocapture bench`.
#[test]
#[ignore = "benchmark: run explicitly in release"]
fn bench_encode_4096() {
    use super::{Bc6hOptions, BcJob};
    let gpu = gpu();
    let mut c = compressor();
    let size = 4096;
    let image = skin(size, size);
    let hdr = hdr_sky(size, size);
    let blocks = u64::from(size / 4) * u64::from(size / 4);
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench blocks"),
        size: blocks * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let ldr = super::offline::upload(gpu, &image, 1).expect("upload");
    let float = super::offline::upload(gpu, &hdr, 1).expect("upload");
    let ldr_view = ldr.create_view(&wgpu::TextureViewDescriptor::default());
    let float_view = float.create_view(&wgpu::TextureViewDescriptor::default());
    let mut report = format!("encode 4096x4096 on {}:\n", gpu.adapter_info().name);
    let mut run = |label: String, record: &mut dyn FnMut(&mut Compressor, &mut wgpu::CommandEncoder)| {
        let mut times = Vec::new();
        for i in 0..4 {
            c.begin();
            let mut encoder =
                gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench") });
            record(&mut c, &mut encoder);
            let start = std::time::Instant::now();
            gpu.queue.submit([encoder.finish()]);
            humanart_gpu::device::wait_for(&gpu.device, None, humanart_gpu::device::GPU_WAIT_LIMIT, "the queued work")
                .expect("poll");
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let _ = writeln!(report, "  {label}: {mean:.1} ms ({:.1} Mtexel/s)", f64::from(size * size) / mean / 1000.0);
    };
    for q in Quality::ALL {
        run(format!("BC7 {q:?}"), &mut |c, e| {
            let job = BcJob { source: &ldr_view, width: size, height: size, out: &out, out_block_offset: 0 };
            c.encode_bc7(gpu, e, &job, &Bc7Options { quality: q, weights: [1.0; 4] }).expect("records");
        });
        run(format!("BC6H UF16 {q:?}"), &mut |c, e| {
            let job = BcJob { source: &float_view, width: size, height: size, out: &out, out_block_offset: 0 };
            c.encode_bc6h(gpu, e, &job, &Bc6hOptions { quality: q, signed: false }).expect("records");
        });
    }
    println!("{report}");
}

/// Records one profiled frame with `record`, submits it and returns its timed scopes (name, GPU
/// milliseconds) once the profiler's asynchronous readback delivered them; `None` marks a pass the
/// adapter could not time.
fn profiled_frame(
    gpu: &Gpu,
    c: &mut Compressor,
    profiler: &mut GpuProfiler,
    record: &mut dyn FnMut(&mut Compressor, &mut GpuProfiler, &mut wgpu::CommandEncoder),
) -> Vec<(&'static str, Option<f64>)> {
    profiler.begin_frame();
    c.begin();
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("profiled") });
    record(c, profiler, &mut encoder);
    profiler.end_frame(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    profiler.after_submit();
    // The test waits for the GPU; the profiler itself only maps its staging buffer and reports
    // from `collect` once the map has finished.
    for _ in 0..1000 {
        humanart_gpu::device::wait_for(&gpu.device, None, humanart_gpu::device::GPU_WAIT_LIMIT, "the queued work")
            .expect("poll");
        if profiler.collect() {
            break;
        }
    }
    profiler.latest().scopes.iter().map(|s| (s.name, s.gpu_nanos.map(|n| n as f64 / 1e6))).collect()
}

#[test]
fn perf_r6_compression_passes_are_timed_with_timestamp_queries() {
    // [PERF-R6] the encoder and mip passes record timestamp scopes; the profiled encode writes the
    // same blocks as the plain one.
    use super::{Bc6hOptions, BcJob, MipContent, MipJob, SCOPE_BC6H, SCOPE_BC7, SCOPE_MIP, create_work_texture};
    let gpu = gpu();
    let mut c = compressor();
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut profiler = GpuProfiler::new(gpu, &mut resources, 2, 16).expect("profiler");
    assert!(profiler.has_timestamps(), "this machine's adapter offers timestamp queries");
    let size = 256;
    let image = skin(size, size);
    let levels = mip::level_count(size, size);
    let texture = super::offline::upload(gpu, &image, levels).expect("upload");
    let work = create_work_texture(gpu, size, size).expect("work chain");
    let float = super::offline::upload(gpu, &hdr_sky(size, size), 1).expect("upload");
    let view = texture.create_view(&wgpu::TextureViewDescriptor { mip_level_count: Some(1), ..Default::default() });
    let float_view = float.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes = u64::from(size / 4) * u64::from(size / 4) * 16;
    let buffer = |label| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let (timed, plain, hdr) = (buffer("timed"), buffer("plain"), buffer("hdr"));
    let options = Bc7Options { quality: Quality::Fast, weights: [1.0; 4] };
    let hdr_options = Bc6hOptions { quality: Quality::Fast, signed: false };
    let scopes = profiled_frame(gpu, &mut c, &mut profiler, &mut |c, p, e| {
        let job = BcJob { source: &view, width: size, height: size, out: &timed, out_block_offset: 0 };
        c.encode_bc7_profiled(gpu, e, &job, &options, p).expect("records");
        let job = BcJob { source: &float_view, width: size, height: size, out: &hdr, out_block_offset: 0 };
        c.encode_bc6h_profiled(gpu, e, &job, &hdr_options, p).expect("records");
        let job = MipJob { texture: &texture, work: &work, content: MipContent::from_color_space(image.color_space) };
        c.generate_mips_profiled(gpu, e, &job, p).expect("records");
    });
    let names: Vec<&str> = scopes.iter().map(|s| s.0).collect();
    let mut expected = vec![SCOPE_BC7, SCOPE_BC6H];
    expected.extend(std::iter::repeat_n(SCOPE_MIP, levels as usize - 1));
    assert_eq!(names, expected);
    assert!(scopes.iter().all(|s| s.1.is_some()), "every pass is timed: {scopes:?}");
    assert!(scopes[0].1.unwrap() > 0.0 && scopes[1].1.unwrap() > 0.0, "{scopes:?}");
    c.begin();
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("plain") });
    let job = BcJob { source: &view, width: size, height: size, out: &plain, out_block_offset: 0 };
    c.encode_bc7(gpu, &mut encoder, &job, &options).expect("records");
    gpu.queue.submit([encoder.finish()]);
    let a = super::offline::read_buffer_blocking(gpu, &timed, bytes).expect("read");
    let b = super::offline::read_buffer_blocking(gpu, &plain, bytes).expect("read");
    assert_eq!(a, b, "the profiled encode writes the same blocks");
    profiler.destroy(&mut resources);
}

/// `[AC-WP28-2]` `[PERF-R6]` GPU time of every texture pass at 4096 x 4096 from timestamp queries
/// (median of 5 frames after a warm-up), next to the wall time of the submission. Run with
/// `cargo test -p humanart-render --release --lib -- --ignored --nocapture gpu_timestamps`.
#[test]
#[ignore = "benchmark: run explicitly in release"]
fn bench_gpu_timestamps_4096() {
    use super::{Bc6hOptions, BcJob, MipContent, MipJob, create_work_texture};
    let gpu = gpu();
    let mut c = compressor();
    let mut resources = Resources::new(gpu, Budget::UNLIMITED);
    let mut profiler = GpuProfiler::new(gpu, &mut resources, 2, 16).expect("profiler");
    assert!(profiler.has_timestamps(), "this machine's adapter offers timestamp queries");
    let size = 4096;
    let image = skin(size, size);
    let hdr = hdr_sky(size, size);
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench blocks"),
        size: u64::from(size / 4) * u64::from(size / 4) * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let levels = mip::level_count(size, size);
    let ldr = super::offline::upload(gpu, &image, levels).expect("upload");
    let float = super::offline::upload(gpu, &hdr, levels).expect("upload");
    let work = create_work_texture(gpu, size, size).expect("work chain");
    let level0 = wgpu::TextureViewDescriptor { mip_level_count: Some(1), ..Default::default() };
    let (ldr_view, float_view) = (ldr.create_view(&level0), float.create_view(&level0));
    let mut report = format!("GPU timestamps, 4096x4096, {}:\n", gpu.adapter_info().name);
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut run =
        |label: String, record: &mut dyn FnMut(&mut Compressor, &mut GpuProfiler, &mut wgpu::CommandEncoder)| {
            let (mut gpu_ms, mut wall_ms) = (Vec::new(), Vec::new());
            for i in 0..6 {
                let start = std::time::Instant::now();
                let scopes = profiled_frame(gpu, &mut c, &mut profiler, record);
                let wall = start.elapsed().as_secs_f64() * 1e3;
                if i > 0 {
                    gpu_ms.push(scopes.iter().filter_map(|s| s.1).sum::<f64>());
                    wall_ms.push(wall);
                }
            }
            let g = median(gpu_ms);
            let _ = writeln!(
                report,
                "  {label}: GPU {g:.2} ms ({:.0} Mtexel/s), wall {:.1} ms",
                f64::from(size * size) / g / 1000.0,
                median(wall_ms)
            );
        };
    for q in Quality::ALL {
        run(format!("BC7 {q:?}"), &mut |c, p, e| {
            let job = BcJob { source: &ldr_view, width: size, height: size, out: &out, out_block_offset: 0 };
            c.encode_bc7_profiled(gpu, e, &job, &Bc7Options { quality: q, weights: [1.0; 4] }, p).expect("records");
        });
        run(format!("BC6H UF16 {q:?}"), &mut |c, p, e| {
            let job = BcJob { source: &float_view, width: size, height: size, out: &out, out_block_offset: 0 };
            c.encode_bc6h_profiled(gpu, e, &job, &Bc6hOptions { quality: q, signed: false }, p).expect("records");
        });
    }
    run(format!("mip chain RGBA8 sRGB, {} levels", levels - 1), &mut |c, p, e| {
        let job = MipJob { texture: &ldr, work: &work, content: MipContent::from_color_space(image.color_space) };
        c.generate_mips_profiled(gpu, e, &job, p).expect("records");
    });
    run(format!("mip chain RGBA16F linear, {} levels", levels - 1), &mut |c, p, e| {
        let job = MipJob { texture: &float, work: &work, content: MipContent::from_color_space(hdr.color_space) };
        c.generate_mips_profiled(gpu, e, &job, p).expect("records");
    });
    println!("{report}");
    profiler.destroy(&mut resources);
}
