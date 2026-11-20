//! Blocking texture reads for stills, the command-line tool and tests (native only: frame code
//! reads back through the asynchronous ring `[PERF-R3]`).

use std::time::Duration;

use humanart_gpu::readback::ReadbackRing;
use humanart_gpu::resources::{Resources, TextureId};
use humanart_math::f16_bits_to_f32;

use crate::error::RenderError;

/// How long a blocking read waits for the GPU before it gives up with
/// [`RenderError::GpuTimeout`]: a slow, shared or lost GPU returns an error instead of hanging.
pub const GPU_WAIT_LIMIT: Duration = Duration::from_secs(60);

/// Waits until `submission` finished on the GPU, at most `limit` `[ERR-05]`. `what` names the
/// wait in the error.
pub fn wait_for_submission(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    limit: Duration,
    what: &'static str,
) -> Result<(), RenderError> {
    match device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(limit) }) {
        Ok(_) => Ok(()),
        Err(wgpu::PollError::Timeout) => {
            Err(RenderError::GpuTimeout { what, limit_ms: u64::try_from(limit.as_millis()).unwrap_or(u64::MAX) })
        }
        Err(e) => Err(humanart_gpu::GpuError::from(e).into()),
    }
}

/// The pixels of one layer of mip 0 of `texture`, rows tightly packed, with its size. Waits at
/// most [`GPU_WAIT_LIMIT`] for the GPU.
pub fn read_texture(
    resources: &mut Resources,
    texture: TextureId,
    layer: u32,
) -> Result<(u32, u32, Vec<u8>), RenderError> {
    let raw =
        resources.texture(texture).ok_or(humanart_gpu::GpuError::Stale { what: "texture to read" })?.raw().clone();
    let size = raw.size();
    let block = raw
        .format()
        .block_copy_size(None)
        .ok_or(RenderError::InvalidParameter { what: "a texture format that cannot be copied whole" })?;
    let row = u64::from(size.width * block).next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let mut ring = ReadbackRing::new(resources, 1, row * u64::from(size.height), "capture")?;
    let device = resources.device().clone();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
    let ticket = ring.copy_texture(&mut encoder, &raw, 0, [0, 0, layer], [size.width, size.height, 1], 0);
    let ticket = match ticket {
        Ok(t) => t,
        Err(e) => {
            ring.destroy(resources);
            return Err(e.into());
        }
    };
    let submission = resources.queue().submit([encoder.finish()]);
    ring.after_submit();
    // The bounded wait finishes the copy and runs the map callback, so the ring's own (unbounded)
    // wait below never blocks.
    if let Err(e) = wait_for_submission(&device, submission, GPU_WAIT_LIMIT, "texture readback") {
        ring.destroy(resources);
        return Err(e);
    }
    let mut out = vec![0_u8; (size.width * block * size.height) as usize];
    let result = ring.read_blocking(&device, ticket, &mut out);
    ring.destroy(resources);
    result?;
    Ok((size.width, size.height, out))
}

/// RGBA f16 texels as f32.
pub fn rgba16f_to_f32(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|t| {
            let h = |i: usize| f16_bits_to_f32(u16::from_le_bytes([t[i], t[i + 1]]));
            [h(0), h(2), h(4), h(6)]
        })
        .collect()
}

/// An 8-bit RGBA image (sRGB-encoded) as a PNG file's bytes.
pub fn encode_png_rgba8(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, RenderError> {
    let image = humanart_io::Image::new(
        humanart_io::Format::Png,
        width,
        height,
        4,
        humanart_io::ColorSpace::Srgb,
        humanart_io::Samples::U8(rgba.to_vec()),
    )?;
    Ok(humanart_io::png::encode(&image)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_texels_decode() {
        let one = humanart_math::f32_to_f16_bits(1.0).to_le_bytes();
        let half = humanart_math::f32_to_f16_bits(0.5).to_le_bytes();
        let bytes = [one, half, one, half].concat();
        assert_eq!(rgba16f_to_f32(&bytes), vec![[1.0, 0.5, 1.0, 0.5]]);
    }

    #[test]
    fn png_round_trips() {
        let png = encode_png_rgba8(2, 1, &[255, 0, 0, 255, 0, 255, 0, 128]).unwrap();
        let back = humanart_io::png::decode(&png, &humanart_io::Limits::default()).unwrap();
        assert_eq!(back.as_u8().unwrap(), &[255, 0, 0, 255, 0, 255, 0, 128]);
    }
}
