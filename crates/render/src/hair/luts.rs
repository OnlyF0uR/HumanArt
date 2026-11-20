//! The fibre tables on the GPU `[HR-02]`: the far-field azimuthal table as a 3D texture and the
//! dual-scattering table as a two-layer 2D array (both `rgba16float`, linear filtering), built
//! from [`FibreParams`] by [`super::bsdf`]. Rebuilt only when the roughness or cuticle angle
//! changes (tens of milliseconds of CPU work), never per frame.

use humanart_gpu::resources::{Category, Resources, SamplerKey, Tag, TextureDesc, TextureId};
use humanart_math::f32_to_f16_bits;

use super::HairError;
use super::bsdf::{AzimuthalTable, DualTable, FibreParams, PHI_SIZE, SIGMA_SIZE, THETA_SIZE};

/// The texel format of both tables.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The fibre tables of one [`FibreParams`].
#[derive(Debug)]
pub struct HairLuts {
    params: FibreParams,
    azimuthal: TextureId,
    dual: TextureId,
    azimuthal_view: wgpu::TextureView,
    dual_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

/// The sampler of the tables: linear, clamped to the edge.
pub fn lut_sampler_key() -> SamplerKey {
    SamplerKey::LINEAR_CLAMP
}

fn halves(values: impl IntoIterator<Item = f32>, out: &mut Vec<u8>) {
    for v in values {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
}

impl HairLuts {
    /// Integrates the tables for `params` and uploads them.
    pub fn new(resources: &mut Resources, params: &FibreParams) -> Result<Self, HairError> {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let azimuthal_desc = TextureDesc {
            width: PHI_SIZE as u32,
            height: THETA_SIZE as u32,
            depth_or_layers: SIGMA_SIZE as u32,
            mip_levels: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: FORMAT,
            usage,
            array: false,
            tag: Tag::shared(Category::Hair, "hair azimuthal table"),
        };
        let (azimuthal, _) = resources.create_texture(&azimuthal_desc)?;
        let dual_desc = TextureDesc::array(
            THETA_SIZE as u32,
            SIGMA_SIZE as u32,
            2,
            FORMAT,
            usage,
            Tag::shared(Category::Hair, "hair dual-scattering table"),
        );
        let dual = match resources.create_texture(&dual_desc) {
            Ok((id, _)) => id,
            Err(e) => {
                let _ = resources.destroy_texture(azimuthal);
                return Err(e.into());
            }
        };
        let view_of = |resources: &Resources, id: TextureId, dimension| {
            resources.texture(id).map(|t| {
                t.raw().create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hair table"),
                    dimension: Some(dimension),
                    ..Default::default()
                })
            })
        };
        let (Some(azimuthal_view), Some(dual_view)) = (
            view_of(resources, azimuthal, wgpu::TextureViewDimension::D3),
            view_of(resources, dual, wgpu::TextureViewDimension::D2Array),
        ) else {
            let _ = resources.destroy_texture(azimuthal);
            let _ = resources.destroy_texture(dual);
            return Err(HairError::Invalid { what: "hair table textures vanished" });
        };
        let sampler = resources.sampler(&lut_sampler_key())?.clone();
        let mut luts = Self { params: *params, azimuthal, dual, azimuthal_view, dual_view, sampler };
        if let Err(e) = luts.upload(resources, params) {
            luts.destroy(resources);
            return Err(e);
        }
        Ok(luts)
    }

    /// Rebuilds the tables when `params` differ from the current ones. Returns whether it did.
    pub fn set_params(&mut self, resources: &mut Resources, params: &FibreParams) -> Result<bool, HairError> {
        if *params == self.params {
            return Ok(false);
        }
        self.upload(resources, params)?;
        Ok(true)
    }

    fn upload(&mut self, resources: &mut Resources, params: &FibreParams) -> Result<(), HairError> {
        let table = AzimuthalTable::build(params).map_err(HairError::Fibre)?;
        let dual = DualTable::build(&table);
        let slice = PHI_SIZE * THETA_SIZE;
        let mut bytes = Vec::with_capacity(slice * 8);
        for (z, chunk) in table.values.chunks_exact(slice).enumerate() {
            bytes.clear();
            halves(chunk.iter().flat_map(|v| *v), &mut bytes);
            resources.write_texture_level(self.azimuthal, 0, z as u32, &bytes)?;
        }
        bytes.clear();
        halves(dual.cells.iter().flat_map(|c| [c.a_f, c.big_a_b, c.delta_b, c.sigma_b2]), &mut bytes);
        resources.write_texture_level(self.dual, 0, 0, &bytes)?;
        bytes.clear();
        halves(dual.cells.iter().flat_map(|c| [c.beta_f2, c.alpha_f, c.a_b, 0.0]), &mut bytes);
        resources.write_texture_level(self.dual, 0, 1, &bytes)?;
        self.params = *params;
        Ok(())
    }

    /// The settings the tables hold.
    pub fn params(&self) -> &FibreParams {
        &self.params
    }

    /// The azimuthal table (3D view).
    pub fn azimuthal_view(&self) -> &wgpu::TextureView {
        &self.azimuthal_view
    }

    /// The dual-scattering table (2D array view).
    pub fn dual_view(&self) -> &wgpu::TextureView {
        &self.dual_view
    }

    /// The linear clamp sampler of both.
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Releases the textures.
    pub fn destroy(&mut self, resources: &mut Resources) {
        let _ = resources.destroy_texture(self.azimuthal);
        let _ = resources.destroy_texture(self.dual);
    }
}
