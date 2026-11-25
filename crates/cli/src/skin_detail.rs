//! `humanart-cli skin-detail` (`todo.md` 1, item 4): the photographed skin colour detail tile
//! (`humanart_model::skin_appearance::albedo_detail`) from a square patch of a scan's albedo: read
//! the PNG, take the patch (sRGB to linear), high-pass it, make it periodic, write it as RGBA8 log
//! ratios. The shipped tile `data/template/skin/albedo-detail.png` was made with
//!
//! ```sh
//! humanart-cli skin-detail --albedo HSR0015-Body-023_LOD0_u0_v0_diffuse.png --rect 8720,3860,256 --blur-px 40 --out data/template/skin/albedo-detail.png
//! ```
//!
//! (HSRD-100, CC BY 4.0: cheek skin of HSR0015-Body-023; `docs/data-provenance.md`).

use std::path::PathBuf;

use humanart_io::Limits;
use humanart_math::color::transfer::srgb_to_linear_f32;
use humanart_model::skin_appearance::albedo_detail::{MAX_SIDE, high_pass, make_periodic, to_rgba8};
use humanart_render::capture::encode_png_rgba8;

use crate::args::{CommandSpec, OptSpec, Parsed, UsageError};
use crate::error::CliError;

const OPTIONS: [OptSpec; 5] = [
    OptSpec { name: "albedo", value: Some("PNG"), default: None, help: "The scan's colour texture (8-bit sRGB)." },
    OptSpec {
        name: "rect",
        value: Some("X,Y,SIDE"),
        default: None,
        help: "The square patch of skin: left, top and side in texels (side a power of two up to 1024).",
    },
    OptSpec {
        name: "blur-px",
        value: Some("PX"),
        default: Some("40"),
        help: "Standard deviation of the high-pass blur, texels (the variation below it is kept).",
    },
    OptSpec { name: "out", value: Some("PNG"), default: Some("albedo-detail.png"), help: "The tile to write." },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `skin-detail` command.
pub static SKIN_DETAIL: CommandSpec = CommandSpec {
    name: "skin-detail",
    summary: "Makes the photographed skin colour detail tile from a patch of a scan's albedo.",
    options: &OPTIONS,
};

/// Largest albedo file read, bytes.
const MAX_ALBEDO_BYTES: u64 = 1 << 30;

/// What `skin-detail` was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinDetailJob {
    /// The scan albedo.
    pub albedo: PathBuf,
    /// Left, top and side of the patch, texels.
    pub rect: [u32; 3],
    /// Blur standard deviation, texels.
    pub blur: f32,
    /// The tile to write.
    pub out: PathBuf,
}

impl SkinDetailJob {
    /// The job of a parsed command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for a missing albedo, a malformed rectangle or blur.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let usage = |m: &str| CliError::Usage(UsageError(m.to_owned()));
        let albedo = p.get("albedo").ok_or_else(|| usage("--albedo is required"))?;
        let rect: Vec<u32> = p
            .get("rect")
            .ok_or_else(|| usage("--rect is required"))?
            .split(',')
            .map(|v| v.trim().parse::<u32>().map_err(|_| usage("--rect: three whole numbers X,Y,SIDE")))
            .collect::<Result<_, _>>()?;
        let [x, y, side] = <[u32; 3]>::try_from(rect).map_err(|_| usage("--rect: three whole numbers X,Y,SIDE"))?;
        if !side.is_power_of_two() || !(16..=MAX_SIDE).contains(&side) {
            return Err(usage("--rect: the side must be a power of two from 16 to 1024"));
        }
        let blur: f32 = p.get("blur-px").unwrap_or("40").parse().map_err(|_| usage("--blur-px: a number"))?;
        if !(blur > 0.5 && blur < 1000.0) {
            return Err(usage("--blur-px: between 0.5 and 1000"));
        }
        Ok(Self {
            albedo: PathBuf::from(albedo),
            rect: [x, y, side],
            blur,
            out: PathBuf::from(p.get("out").unwrap_or("albedo-detail.png")),
        })
    }
}

/// Runs `job` and returns the standard deviation of the log ratio per channel.
///
/// # Errors
/// [`CliError::Io`] for the files, [`CliError::Data`] for an unusable texture or a patch outside it.
pub fn run(job: &SkinDetailJob) -> Result<[f32; 3], CliError> {
    let size = std::fs::metadata(&job.albedo).map_err(|e| CliError::io(&job.albedo, e))?.len();
    if size > MAX_ALBEDO_BYTES {
        return Err(CliError::Data(format!("{}: larger than 1 GiB", job.albedo.display())));
    }
    let bytes = std::fs::read(&job.albedo).map_err(|e| CliError::io(&job.albedo, e))?;
    let limits = Limits { max_dimension: 16_384, max_pixels: 1 << 29, max_bytes: 1 << 31, ..Limits::DEFAULT };
    let image = humanart_io::png::decode(&bytes, &limits)
        .map_err(|e| CliError::Data(format!("{}: {e}", job.albedo.display())))?;
    drop(bytes);
    let samples = image.as_u8().ok_or_else(|| CliError::Data("the albedo must be an 8-bit image".into()))?;
    let channels = usize::from(image.channels);
    if channels < 3 {
        return Err(CliError::Data("the albedo must have colour channels".into()));
    }
    let [x0, y0, side] = job.rect;
    if x0.saturating_add(side) > image.width || y0.saturating_add(side) > image.height {
        return Err(CliError::Data("the patch lies outside the albedo".into()));
    }
    let width = image.width as usize;
    let s = side as usize;
    let mut rgb = Vec::with_capacity(s * s);
    for y in 0..s {
        for x in 0..s {
            let i = ((y0 as usize + y) * width + x0 as usize + x) * channels;
            let px =
                samples.get(i..i + 3).ok_or_else(|| CliError::Data("the albedo is shorter than its size".into()))?;
            rgb.push([0, 1, 2].map(|c| srgb_to_linear_f32(f32::from(px[c]) / 255.0)));
        }
    }
    let detail = make_periodic(&high_pass(&rgb, s, job.blur).map_err(|e| CliError::Data(e.to_string()))?, s);
    let sd = [0, 1, 2].map(|c| (detail.iter().map(|p| p[c] * p[c]).sum::<f32>() / detail.len() as f32).sqrt());
    let png = encode_png_rgba8(side, side, &to_rgba8(&detail)).map_err(|e| CliError::Data(e.to_string()))?;
    if let Some(dir) = job.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
    }
    std::fs::write(&job.out, png).map_err(|e| CliError::io(&job.out, e))?;
    Ok(sd)
}
