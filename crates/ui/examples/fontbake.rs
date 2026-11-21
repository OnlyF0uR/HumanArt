//! Offline font baking `[UI-03]`: TrueType fonts to `.anfont` distance-field atlases.
//!
//! ```text
//! cargo run --release -p humanart-ui --example fontbake -- <fonts dir>
//!     bakes the shipped faces of <fonts dir> (normally data/fonts) into <fonts dir>/<face>.anfont
//! cargo run --release -p humanart-ui --example fontbake -- --font <in.ttf> <out.anfont> [text|icons]
//!     bakes one font: the Latin set (text, the default) or every mapped character (icons)
//! ```
//!
//! `tools/fontbake/bake.sh` runs the first form. Each baked file is reported with its glyph count,
//! kerning pairs, pages and size. `humanart-cli fontbake` does the same from the command line
//! (`--dir`, or `--font` with `--out` and `--kind`), through the same library functions.

use std::path::{Path, PathBuf};
use std::time::Instant;

use humanart_ui::fontbake::{bake_shipped, source_file};
use humanart_ui::theme::FontFace;

fn bake_one(input: &Path, output: &Path, icons: bool) -> Result<(), String> {
    let ttf = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let start = Instant::now();
    let baked = bake_shipped(&ttf, icons).map_err(|e| format!("{}: {e}", input.display()))?;
    let bytes = baked.to_bytes();
    std::fs::write(output, &bytes).map_err(|e| format!("{}: {e}", output.display()))?;
    println!(
        "{} -> {}: {} glyphs, {} kerning pairs, {} page(s) of {}^2, {} bytes, {:.2} s",
        input.display(),
        output.display(),
        baked.glyphs.len(),
        baked.kerning.len(),
        baked.pages,
        baked.page_size,
        bytes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, input, output, rest @ ..] if flag == "--font" => {
            let icons = match rest {
                [] => false,
                [kind] if kind == "text" => false,
                [kind] if kind == "icons" => true,
                _ => return Err("the kind is `text` or `icons`".into()),
            };
            bake_one(Path::new(input), Path::new(output), icons)
        }
        [dir] => {
            let dir = PathBuf::from(dir);
            for face in FontFace::ALL {
                let output = dir.join(format!("{}.anfont", face.name()));
                bake_one(&dir.join(source_file(face)), &output, face == FontFace::Icons)?;
            }
            Ok(())
        }
        _ => Err("usage: fontbake <fonts dir> | fontbake --font <in.ttf> <out.anfont> [text|icons]".into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("fontbake: {e}");
        std::process::exit(1);
    }
}
