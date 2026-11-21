//! Blocking whole-image helpers over [`Compressor`] for the command-line tool, the exporter and
//! tests (native only: they wait for the GPU, which a frame never does, `[PERF-R3]`).
//!
//! [`Compressor::compress_blocking`] uploads an [`Image`], optionally generates its mips on the GPU,
//! encodes every level and reads the blocks back; [`to_ktx2`] wraps the result for
//! `humanart_io::ktx2::write`.

use humanart_gpu::{Gpu, GpuError};
use humanart_io::bc::BcFormat;
use humanart_io::ktx2::{self, FormatInfo};
use humanart_io::{ColorSpace, Format, Image, Samples};
use humanart_math::{f16_bits_to_f32, f32_to_f16_bits};

use super::{
    Bc6hOptions, Bc7Options, BcJob, CompressError, Compressor, MipContent, MipJob, create_work_texture, level_view,
};

/// What [`Compressor::compress_blocking`] produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    /// The block format. BC7 takes 8-bit images, BC6H float images.
    pub format: BcFormat,
    /// Generate the full mip chain on the GPU (otherwise level 0 only).
    pub mips: bool,
    /// BC7 settings.
    pub bc7: Bc7Options,
    /// BC6H settings (`signed` is taken from `format`).
    pub bc6h: Bc6hOptions,
    /// Also read back the uncompressed levels (to measure quality).
    pub keep_levels: bool,
}

impl Request {
    /// Defaults for `format`: full mip chain, normal quality.
    pub fn new(format: BcFormat) -> Self {
        Self { format, mips: true, bc7: Bc7Options::default(), bc6h: Bc6hOptions::default(), keep_levels: false }
    }
}

/// An encoded mip chain.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// The block format.
    pub format: BcFormat,
    /// Level 0 width.
    pub width: u32,
    /// Level 0 height.
    pub height: u32,
    /// The colour space of the source.
    pub color_space: ColorSpace,
    /// The block stream of every level, level 0 first.
    pub blocks: Vec<Vec<u8>>,
    /// The uncompressed levels as the encoder saw them (RGBA, 8-bit or half), when requested.
    pub levels: Vec<Image>,
}

/// A texture holding `image` in level 0 (`Rgba8Unorm` for 8-bit images, `Rgba16Float` for float
/// images), with room for `levels` levels and every usage the helpers need.
pub fn upload(gpu: &Gpu, image: &Image, levels: u32) -> Result<wgpu::Texture, CompressError> {
    let channels = image.channels as usize;
    let expand = |src: &[u8], one: u8| -> Vec<u8> {
        let mut out = Vec::with_capacity(image.pixel_count() * 4);
        for p in src.chunks_exact(channels) {
            match channels {
                1 => out.extend([p[0], p[0], p[0], one]),
                2 => out.extend([p[0], p[0], p[0], p[1]]),
                3 => out.extend([p[0], p[1], p[2], one]),
                _ => out.extend([p[0], p[1], p[2], p[3]]),
            }
        }
        out
    };
    let (format, bytes) = match &image.samples {
        Samples::U8(v) => (wgpu::TextureFormat::Rgba8Unorm, expand(v, 255)),
        Samples::F16(v) => {
            let halves: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
            let one = f32_to_f16_bits(1.0).to_le_bytes();
            (wgpu::TextureFormat::Rgba16Float, expand_wide(&halves, channels, one, image.pixel_count()))
        }
        Samples::F32(v) => {
            let halves: Vec<u8> = v.iter().flat_map(|&f| f32_to_f16_bits(f).to_le_bytes()).collect();
            let one = f32_to_f16_bits(1.0).to_le_bytes();
            (wgpu::TextureFormat::Rgba16Float, expand_wide(&halves, channels, one, image.pixel_count()))
        }
        Samples::U16(_) => {
            return Err(CompressError::Invalid {
                what: "16-bit unsigned images are not supported; convert to 8-bit or float",
            });
        }
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture source"),
        size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texel = if format == wgpu::TextureFormat::Rgba8Unorm { 4 } else { 8 };
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * texel),
            rows_per_image: Some(image.height),
        },
        wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
    );
    Ok(texture)
}

/// Expands 2-byte samples of `channels` channels to RGBA, `one` for missing alpha.
fn expand_wide(src: &[u8], channels: usize, one: [u8; 2], pixels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels * 8);
    for p in src.chunks_exact(channels * 2) {
        let s = |c: usize| [p[2 * c], p[2 * c + 1]];
        let texel = match channels {
            1 => [s(0), s(0), s(0), one],
            2 => [s(0), s(0), s(0), s(1)],
            3 => [s(0), s(1), s(2), one],
            _ => [s(0), s(1), s(2), s(3)],
        };
        for v in texel {
            out.extend(v);
        }
    }
    out
}

/// Waits for the GPU and reads `size` bytes of a `COPY_SRC` buffer.
pub fn read_buffer_blocking(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, CompressError> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);
    map_blocking(gpu, &staging)
}

/// Maps a `MAP_READ` buffer after the queued work and copies it out.
fn map_blocking(gpu: &Gpu, staging: &wgpu::Buffer) -> Result<Vec<u8>, CompressError> {
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    humanart_gpu::device::wait_for(&gpu.device, None, humanart_gpu::device::GPU_WAIT_LIMIT, "the queued work")?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(GpuError::Map { message: e.to_string() }.into()),
        Err(_) => return Err(GpuError::Map { message: String::from("the map callback never ran") }.into()),
    }
    let bytes = staging.slice(..).get_mapped_range().map_err(|e| GpuError::Map { message: e.to_string() })?.to_vec();
    staging.unmap();
    Ok(bytes)
}

/// Waits for the GPU and reads level `level` of a `COPY_SRC` texture, rows tightly packed.
pub fn read_texture_blocking(gpu: &Gpu, texture: &wgpu::Texture, level: u32) -> Result<Vec<u8>, CompressError> {
    let texel = texture
        .format()
        .block_copy_size(None)
        .ok_or(CompressError::Invalid { what: "the texture format cannot be copied" })?;
    let (w, h) = humanart_io::mip::level_size(texture.width(), texture.height(), level);
    let row = w * texel;
    let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture readback"),
        size: u64::from(padded) * u64::from(h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    let bytes = map_blocking(gpu, &staging)?;
    let mut out = Vec::with_capacity(row as usize * h as usize);
    for y in 0..h as usize {
        let start = y * padded as usize;
        out.extend_from_slice(
            bytes.get(start..start + row as usize).ok_or(CompressError::Invalid { what: "short readback" })?,
        );
    }
    Ok(out)
}

/// A read-back level as an RGBA image.
fn level_image(bytes: Vec<u8>, w: u32, h: u32, float: bool, space: ColorSpace) -> Result<Image, CompressError> {
    let samples = if float {
        Samples::F16(bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    } else {
        Samples::U8(bytes)
    };
    Ok(Image::new(Format::Bc, w, h, 4, space, samples)?)
}

impl Compressor {
    /// Generates the full mip chain of `image` on the GPU and reads every level back as RGBA
    /// (8-bit for 8-bit input, half floats otherwise). Level 0 is the uploaded input.
    pub fn mips_blocking(&mut self, gpu: &Gpu, image: &Image) -> Result<Vec<Image>, CompressError> {
        let levels = humanart_io::mip::level_count(image.width, image.height);
        let texture = upload(gpu, image, levels)?;
        if levels > 1 {
            let work = create_work_texture(gpu, image.width, image.height)?;
            self.begin();
            let mut encoder =
                gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mips") });
            self.generate_mips(
                gpu,
                &mut encoder,
                &MipJob { texture: &texture, work: &work, content: MipContent::from_color_space(image.color_space) },
            )?;
            gpu.queue.submit([encoder.finish()]);
        }
        let float = texture.format() == wgpu::TextureFormat::Rgba16Float;
        let mut out = Vec::with_capacity(levels as usize);
        for level in 0..levels {
            let (w, h) = humanart_io::mip::level_size(image.width, image.height, level);
            out.push(level_image(read_texture_blocking(gpu, &texture, level)?, w, h, float, image.color_space)?);
        }
        Ok(out)
    }

    /// Uploads `image`, generates its mips when asked, encodes every level and reads the blocks back.
    pub fn compress_blocking(&mut self, gpu: &Gpu, image: &Image, request: &Request) -> Result<Encoded, CompressError> {
        let bc7 = request.format == BcFormat::Bc7;
        if bc7 != matches!(image.samples, Samples::U8(_)) {
            return Err(CompressError::Invalid { what: "BC7 encodes 8-bit images and BC6H float images" });
        }
        let levels = if request.mips { humanart_io::mip::level_count(image.width, image.height) } else { 1 };
        let texture = upload(gpu, image, levels)?;
        let mut offsets = Vec::with_capacity(levels as usize);
        let mut total = 0_u64;
        for level in 0..levels {
            let (w, h) = humanart_io::mip::level_size(image.width, image.height, level);
            offsets.push(total);
            let (bx, by) = humanart_io::bc::block_dims(w, h);
            total += u64::from(bx) * u64::from(by);
        }
        let first_block = |level: usize| -> Result<u32, CompressError> {
            u32::try_from(offsets[level]).map_err(|_| CompressError::Invalid { what: "too many blocks" })
        };
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc blocks"),
            size: total * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.begin();
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compress") });
        let work = if levels > 1 { Some(create_work_texture(gpu, image.width, image.height)?) } else { None };
        if let Some(work) = &work {
            self.generate_mips(
                gpu,
                &mut encoder,
                &MipJob { texture: &texture, work, content: MipContent::from_color_space(image.color_space) },
            )?;
        }
        for level in 0..levels {
            let (w, h) = humanart_io::mip::level_size(image.width, image.height, level);
            let view = level_view(&texture, level);
            let job =
                BcJob { source: &view, width: w, height: h, out: &out, out_block_offset: first_block(level as usize)? };
            if bc7 {
                self.encode_bc7(gpu, &mut encoder, &job, &request.bc7)?;
            } else {
                let options = Bc6hOptions { signed: request.format == BcFormat::Bc6hSfloat, ..request.bc6h };
                self.encode_bc6h(gpu, &mut encoder, &job, &options)?;
            }
        }
        gpu.queue.submit([encoder.finish()]);
        let bytes = read_buffer_blocking(gpu, &out, total * 16)?;
        let mut blocks = Vec::with_capacity(levels as usize);
        for level in 0..levels as usize {
            let start = offsets[level] as usize * 16;
            let end = offsets.get(level + 1).map_or(bytes.len(), |&o| o as usize * 16);
            blocks.push(bytes.get(start..end).ok_or(CompressError::Invalid { what: "short readback" })?.to_vec());
        }
        let mut kept = Vec::new();
        if request.keep_levels {
            for level in 0..levels {
                let (w, h) = humanart_io::mip::level_size(image.width, image.height, level);
                let bytes = read_texture_blocking(gpu, &texture, level)?;
                kept.push(level_image(bytes, w, h, !bc7, image.color_space)?);
            }
        }
        Ok(Encoded {
            format: request.format,
            width: image.width,
            height: image.height,
            color_space: image.color_space,
            blocks,
            levels: kept,
        })
    }
}

/// Wraps an encoded chain as a KTX2 texture (`VK_FORMAT_BC7_SRGB_BLOCK` for sRGB colour,
/// `BC7_UNORM`, `BC6H_UFLOAT` or `BC6H_SFLOAT` otherwise).
pub fn to_ktx2(encoded: &Encoded) -> Result<ktx2::Texture, CompressError> {
    let vk = match encoded.format {
        BcFormat::Bc7 if encoded.color_space == ColorSpace::Srgb => ktx2::vk::BC7_SRGB,
        BcFormat::Bc7 => ktx2::vk::BC7_UNORM,
        BcFormat::Bc6hUfloat => ktx2::vk::BC6H_UFLOAT,
        BcFormat::Bc6hSfloat => ktx2::vk::BC6H_SFLOAT,
    };
    let format =
        FormatInfo::from_vk(vk).ok_or(CompressError::Invalid { what: "the KTX2 format table lacks a BC format" })?;
    Ok(ktx2::Texture::new_2d(format, encoded.width, encoded.height, encoded.blocks.clone())?)
}

/// Half-float samples to `f32` (for quality measures).
pub fn halves_to_f32(halves: &[u16]) -> Vec<f32> {
    halves.iter().map(|&h| f16_bits_to_f32(h)).collect()
}
