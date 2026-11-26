//! `humanart-cli face-detail` (`todo.md` 1, ADR-060): the regional skin colour detail of one UV
//! tile (`humanart_model::skin_appearance::albedo_detail::regional_detail`) from a scan's albedo
//! transferred into the template's layout by `humanart-cli fit-scan`.
//!
//! Inputs are that tile of the fit's `albedo-atlas.png` and `error-atlas.png` (the atlas holds
//! the fitted tiles side by side; crop the tile's square out first). Each is box-filtered to
//! `--size`. A texel is used where the fit found the scan within `--max-error-mm` (the error map's
//! colours, `scan::error_colour`) and the template's region there is skin to keep: forehead,
//! nose without the nostrils, cheeks, malar, chin and philtrum, not the brows, eyes, lids,
//! sockets, lips, temples, jaw (beside the hair and the neck's shadow), ears, hairline, scalp or neck (the scan's makeup, brow hairs, lipstick,
//! hair and collar); and of those, not a texel much darker ([`DARK`]: stray hair, liner) or redder
//! ([`RED`]: lipstick) than the kept skin's median. The shipped `data/template/skin/face-detail.png` was made with
//!
//! ```sh
//! humanart-cli face-detail --albedo tile0.png --error err0.png --size 2048 --sigma-mm 3 --feather-mm 2 --limit 0.12 --out data/template/skin/face-detail.png
//! ```
//!
//! from the fit of HSRD-100 HSR0015-Body-023 (CC BY 4.0; `docs/data-provenance.md`).

use std::path::{Path, PathBuf};

use humanart_io::{Image, Limits};
use humanart_math::color::transfer::srgb_to_linear_f32;
use humanart_model::skin_appearance::albedo_detail::{MAX_REGIONAL_SIDE, regional_detail, to_rgba8};
use humanart_model::skin_appearance::zone_bake::{ZoneWeights, bake_tile};
use humanart_render::capture::encode_png_rgba8;

use crate::args::{CommandSpec, OptSpec, Parsed, UsageError};
use crate::error::CliError;
use crate::render::load_template;

const OPTIONS: [OptSpec; 12] = [
    OptSpec { name: "albedo", value: Some("PNG"), default: None, help: "The tile of the fit's albedo atlas (square)." },
    OptSpec { name: "error", value: Some("PNG"), default: None, help: "The same tile of the fit's error atlas." },
    OptSpec { name: "tile", value: Some("NAME"), default: Some("face"), help: "The template's UV tile it is." },
    OptSpec {
        name: "size",
        value: Some("PX"),
        default: Some("2048"),
        help: "Side of the map written (a power of two).",
    },
    OptSpec {
        name: "sigma-mm",
        value: Some("MM"),
        default: Some("3"),
        help: "Scale of the high-pass: the variation below it is kept, millimetres.",
    },
    OptSpec {
        name: "feather-mm",
        value: Some("MM"),
        default: Some("2"),
        help: "Fade at the edges of what is kept, mm.",
    },
    OptSpec {
        name: "limit",
        value: Some("L"),
        default: Some("0.12"),
        help: "Soft limit of the log ratio (L tanh(l / L)), at most 0.5.",
    },
    OptSpec {
        name: "max-error-mm",
        value: Some("MM"),
        default: Some("1.5"),
        help: "Farthest fit distance of a texel used, millimetres.",
    },
    OptSpec { name: "preview", value: Some("PNG"), default: None, help: "Also write the map with 4x contrast." },
    OptSpec { name: "out", value: Some("PNG"), default: Some("face-detail.png"), help: "The map to write." },
    OptSpec { name: "data", value: Some("DIR"), default: Some("data"), help: "The data folder." },
    OptSpec { name: "json-errors", value: None, default: None, help: "Errors as one JSON line on stderr." },
];

/// The `face-detail` command.
pub static FACE_DETAIL: CommandSpec = CommandSpec {
    name: "face-detail",
    summary: "Makes a UV tile's regional skin colour detail from a fitted scan's transferred albedo.",
    options: &OPTIONS,
};

/// Regions whose skin is kept (with their sub-regions).
pub const KEEP: [&str; 11] = [
    "forehead",
    "nasal-bridge",
    "nasal-tip",
    "ala-L",
    "ala-R",
    "philtrum",
    "chin",
    "cheek-L",
    "cheek-R",
    "malar-L",
    "malar-R",
];
/// A kept texel darker than this share of the kept skin's median luminance is left out (hair,
/// brow hairs, liner, deep shadow).
pub const DARK: f32 = 0.75;
/// A kept texel whose red over green exceeds the kept skin's median by this factor is left out
/// (lipstick, lip colour).
pub const RED: f32 = 1.3;

/// Largest PNG read, bytes.
const MAX_BYTES: u64 = 1 << 30;

/// What `face-detail` was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceDetailJob {
    /// The albedo tile.
    pub albedo: PathBuf,
    /// The error tile.
    pub error: PathBuf,
    /// The UV tile's name.
    pub tile: String,
    /// Side written.
    pub size: u32,
    /// High-pass scale, mm.
    pub sigma_mm: f32,
    /// Edge fade, mm.
    pub feather_mm: f32,
    /// Soft limit of the log ratio.
    pub limit: f32,
    /// Largest fit distance used, mm.
    pub max_error_mm: f32,
    /// Optional contrast preview.
    pub preview: Option<PathBuf>,
    /// The map written.
    pub out: PathBuf,
    /// The data folder.
    pub data: PathBuf,
}

impl FaceDetailJob {
    /// The job of a parsed command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] for missing files or values out of range.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let usage = |m: &str| CliError::Usage(UsageError(m.to_owned()));
        let number = |name: &str, default: &str, lo: f32, hi: f32| -> Result<f32, CliError> {
            let v: f32 = p.get(name).unwrap_or(default).parse().map_err(|_| usage(&format!("--{name}: a number")))?;
            if v > lo && v < hi { Ok(v) } else { Err(usage(&format!("--{name}: between {lo} and {hi}"))) }
        };
        let size = p.number("size", 16, MAX_REGIONAL_SIDE)?;
        if !size.is_power_of_two() {
            return Err(usage("--size: a power of two"));
        }
        Ok(Self {
            albedo: PathBuf::from(p.get("albedo").ok_or_else(|| usage("--albedo is required"))?),
            error: PathBuf::from(p.get("error").ok_or_else(|| usage("--error is required"))?),
            tile: p.get("tile").unwrap_or("face").to_owned(),
            size,
            sigma_mm: number("sigma-mm", "3", 0.1, 500.0)?,
            feather_mm: number("feather-mm", "2", 0.05, 100.0)?,
            limit: number("limit", "0.12", 0.0, 0.5001)?,
            max_error_mm: number("max-error-mm", "1.5", 0.01, 100.0)?,
            preview: p.get("preview").map(PathBuf::from),
            out: PathBuf::from(p.get("out").unwrap_or("face-detail.png")),
            data: PathBuf::from(p.get("data").unwrap_or("data")),
        })
    }
}

fn read_png(path: &Path) -> Result<Image, CliError> {
    let size = std::fs::metadata(path).map_err(|e| CliError::io(path, e))?.len();
    if size > MAX_BYTES {
        return Err(CliError::Data(format!("{}: larger than 1 GiB", path.display())));
    }
    let bytes = std::fs::read(path).map_err(|e| CliError::io(path, e))?;
    let limits = Limits { max_dimension: 16_384, max_pixels: 1 << 28, max_bytes: 1 << 31, ..Limits::DEFAULT };
    humanart_io::png::decode(&bytes, &limits).map_err(|e| CliError::Data(format!("{}: {e}", path.display())))
}

/// The image box-filtered to `size`² as sRGB-encoded floats 0..1 per channel (3 channels).
fn downsample(image: &Image, size: u32, path: &Path) -> Result<Vec<[f32; 3]>, CliError> {
    let samples = image.as_u8().ok_or_else(|| CliError::Data(format!("{}: not an 8-bit image", path.display())))?;
    let channels = usize::from(image.channels);
    if channels < 3 || image.width != image.height || !image.width.is_multiple_of(size) {
        return Err(CliError::Data(format!(
            "{}: a square colour image whose side is a multiple of --size",
            path.display()
        )));
    }
    let f = (image.width / size) as usize;
    let (w, s) = (image.width as usize, size as usize);
    let mut out = vec![[0.0_f32; 3]; s * s];
    let scale = 1.0 / (255.0 * (f * f) as f32);
    for (k, o) in out.iter_mut().enumerate() {
        let (x0, y0) = ((k % s) * f, (k / s) * f);
        let mut sum = [0_u32; 3];
        for y in y0..y0 + f {
            for x in x0..x0 + f {
                let i = (y * w + x) * channels;
                let px =
                    samples.get(i..i + 3).ok_or_else(|| CliError::Data("an image is shorter than its size".into()))?;
                for c in 0..3 {
                    sum[c] += u32::from(px[c]);
                }
            }
        }
        *o = sum.map(|v| v as f32 * scale);
    }
    Ok(out)
}

/// Whether an error-map colour (sRGB 0..1) is a fit distance below `max_mm`: the map runs blue
/// (0), green (1 mm), yellow (2 mm), red (4 mm), magenta (no scan point); red grows with the
/// distance from 1 mm on and is 0.7 linear at 2 mm; black is outside the islands.
fn within(c: [f32; 3], max_mm: f32) -> bool {
    let lin = c.map(srgb_to_linear_f32);
    // Magenta (no scan point) and black (outside the islands).
    if (lin[0] > 0.4 && lin[2] > 0.4) || lin.iter().all(|&v| v < 0.02) {
        return false;
    }
    let mm = if lin[0] <= 1e-3 {
        1.0 - (lin[2] / 0.6).min(1.0)
    } else if lin[1] > 0.3 {
        1.0 + lin[0] / 0.7
    } else {
        4.0
    };
    mm <= max_mm
}

/// Runs `job` and returns the share of texels used and the log ratio's SD per channel there.
///
/// # Errors
/// [`CliError::Io`] for the files, [`CliError::Data`] for unusable images or a template without
/// the tile.
pub fn run(job: &FaceDetailJob) -> Result<(f32, [f32; 3]), CliError> {
    let (template, _) = load_template(&job.data)?;
    let tile = template
        .tiles()
        .iter()
        .position(|t| t.name == job.tile)
        .ok_or_else(|| CliError::Data(format!("the template has no UV tile `{}`", job.tile)))?;
    let tile16 = u16::try_from(tile).map_err(|_| CliError::Data("UV tile index".into()))?;
    let texel_mm = template.mm_per_uv().get(tile).copied().unwrap_or(0.0) / job.size as f32;
    let albedo = downsample(&read_png(&job.albedo)?, job.size, &job.albedo)?;
    let error = downsample(&read_png(&job.error)?, job.size, &job.error)?;
    // The skin to keep, per vertex, rasterised into the tile.
    let regions = template.regions();
    let keep: Vec<u16> = KEEP.iter().filter_map(|n| regions.find(n)).collect();
    let values: Vec<f32> = regions
        .vertex_regions
        .iter()
        .map(|&r| if keep.iter().any(|&k| regions.is_within(r, k)) { 1.0 } else { 0.0 })
        .collect();
    let mesh = template.mesh();
    let mut weights = ZoneWeights::new(mesh.vertex_count());
    if !weights.set_plain(0, &values) {
        return Err(CliError::Data("the template's region map does not match its mesh".into()));
    }
    let masks = bake_tile(mesh, &mesh.triangulate(), &weights, tile16, job.size, 0)
        .map_err(|e| CliError::Data(e.to_string()))?;
    let s = job.size as usize;
    let weight: Vec<f32> = (0..s * s)
        .map(|k| {
            let (x, y) = ((k % s) as u32, (k / s) as u32);
            if within(error[k], job.max_error_mm) { masks.weight(0, x, y) } else { 0.0 }
        })
        .collect();
    let linear: Vec<[f32; 3]> = albedo.iter().map(|c| c.map(srgb_to_linear_f32)).collect();
    // Gates against the kept skin's medians: hair, brows and liner are much darker, lipstick much
    // redder.
    let lum = |c: &[f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let red = |c: &[f32; 3]| c[0] / c[1].max(1e-4);
    let median = |mut v: Vec<f32>| {
        v.sort_unstable_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let kept: Vec<usize> = (0..s * s).filter(|&k| weight[k] > 0.5).collect();
    let lum_median = median(kept.iter().map(|&k| lum(&linear[k])).collect());
    let red_median = median(kept.iter().map(|&k| red(&linear[k])).collect());
    let weight: Vec<f32> = weight
        .iter()
        .zip(&linear)
        .map(|(&w, c)| if lum(c) < DARK * lum_median || red(c) > RED * red_median { 0.0 } else { w })
        .collect();
    let detail = regional_detail(&linear, &weight, s, texel_mm, job.sigma_mm, job.feather_mm, job.limit.min(0.5))
        .map_err(|e| CliError::Data(e.to_string()))?;
    let used: Vec<usize> = (0..s * s).filter(|&k| detail[k] != [0.0; 3]).collect();
    let n = used.len().max(1) as f32;
    let sd = [0, 1, 2].map(|c| (used.iter().map(|&k| detail[k][c].powi(2)).sum::<f32>() / n).sqrt());
    let write = |path: &Path, values: &[[f32; 3]]| -> Result<(), CliError> {
        let png = encode_png_rgba8(job.size, job.size, &to_rgba8(values)).map_err(|e| CliError::Data(e.to_string()))?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| CliError::io(dir, e))?;
        }
        std::fs::write(path, png).map_err(|e| CliError::io(path, e))
    };
    write(&job.out, &detail)?;
    if let Some(preview) = &job.preview {
        let strong: Vec<[f32; 3]> = detail.iter().map(|p| p.map(|v| v * 4.0)).collect();
        write(preview, &strong)?;
    }
    Ok((used.len() as f32 / (s * s) as f32, sd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::error_colour;
    use humanart_math::color::transfer::linear_to_srgb_rgb_f32;

    #[test]
    fn todo_1_error_colours_decode_to_their_distance() {
        let encoded = |mm: f32| linear_to_srgb_rgb_f32(error_colour(mm / 1000.0));
        assert!(within(encoded(0.2), 1.5));
        assert!(within(encoded(1.0), 1.5));
        assert!(within(encoded(1.4), 1.5));
        assert!(!within(encoded(1.8), 1.5));
        assert!(!within(encoded(5.0), 1.5));
        assert!(!within(linear_to_srgb_rgb_f32(error_colour(-1.0)), 1.5), "no scan point");
        assert!(!within([0.0; 3], 1.5), "outside the islands");
    }
}
