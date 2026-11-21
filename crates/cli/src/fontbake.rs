//! `humanart-cli fontbake` `[UI-03]`: TrueType fonts to `.anfont` distance-field atlases, with the
//! library baker `humanart_ui::fontbake`.
//!
//! - `fontbake [--dir DIR]` bakes the five shipped faces from their sources in `DIR` (normally
//!   `data/fonts`: Inter, JetBrains Mono, Lucide) into `DIR/<face>.anfont`;
//! - `fontbake --font IN.ttf --out OUT.anfont [--kind text|icons]` bakes one font: the Latin set
//!   (`text`) or every character the font maps (`icons`).
//!
//! Every font is read and baked before its file is written, so a font that does not bake leaves
//! the existing `.anfont` as it was. Each baked file is reported with its glyph count, kerning
//! pairs, pages and size. The bake is deterministic: the same font gives the same bytes.

use std::path::{Path, PathBuf};
use std::time::Instant;

use humanart_ui::fontbake::{bake_shipped, source_file};
use humanart_ui::theme::FontFace;

use crate::args::{CommandSpec, OptSpec, Parsed, UsageError};
use crate::error::CliError;
use crate::host;

/// The fonts folder when neither `--dir` nor `--font` is given.
pub const DEFAULT_DIR: &str = "data/fonts";
/// The largest TrueType file read (the shipped ones are under 1 MiB).
pub const MAX_FONT_BYTES: u64 = 64 << 20;

const KINDS: [&str; 2] = ["text", "icons"];

const OPTIONS: [OptSpec; 5] = [
    OptSpec {
        name: "dir",
        value: Some("DIR"),
        default: None,
        help: "Fonts folder: bakes the five shipped faces from their sources in it (without --font: data/fonts).",
    },
    OptSpec { name: "font", value: Some("FILE"), default: None, help: "One TrueType font to bake (with --out)." },
    OptSpec { name: "out", value: Some("FILE"), default: None, help: "The .anfont file --font is baked into." },
    OptSpec {
        name: "kind",
        value: Some("KIND"),
        default: None,
        help: "With --font: text (the Latin set, the default) or icons (every mapped character).",
    },
    host::JSON_ERRORS,
];

/// The `fontbake` command.
pub static FONTBAKE: CommandSpec = CommandSpec {
    name: "fontbake",
    summary: "Bakes TrueType fonts into the UI's .anfont distance-field atlases.",
    options: &OPTIONS,
};

/// What `fontbake` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FontbakeJob {
    /// The five shipped faces of a fonts folder.
    Shipped {
        /// The folder holding the sources and receiving the `.anfont` files.
        dir: PathBuf,
    },
    /// One font.
    One {
        /// The TrueType file.
        font: PathBuf,
        /// The `.anfont` file to write.
        out: PathBuf,
        /// Every mapped character at the icon options instead of the Latin set.
        icons: bool,
    },
}

impl FontbakeJob {
    /// The job of a parsed `fontbake` command line.
    ///
    /// # Errors
    /// [`CliError::Usage`] when `--font` and `--out` do not come together, `--dir` is mixed with
    /// them, `--kind` is given without `--font` or is not a kind.
    pub fn from_args(p: &Parsed) -> Result<Self, CliError> {
        let usage = |text: &str| CliError::Usage(UsageError(format!("fontbake: {text}")));
        match (p.get("font"), p.get("out")) {
            (Some(font), Some(out)) => {
                if p.get("dir").is_some() {
                    return Err(usage("--dir bakes the shipped faces; it does not go with --font"));
                }
                let icons = p.get("kind").is_some() && p.choice("kind", &KINDS)? == 1;
                Ok(Self::One { font: PathBuf::from(font), out: PathBuf::from(out), icons })
            }
            (Some(_), None) => Err(usage("--font needs --out")),
            (None, Some(_)) => Err(usage("--out needs --font")),
            (None, None) => {
                if p.get("kind").is_some() {
                    return Err(usage("--kind needs --font (the shipped faces have their own kinds)"));
                }
                Ok(Self::Shipped { dir: PathBuf::from(p.get("dir").unwrap_or(DEFAULT_DIR)) })
            }
        }
    }
}

/// One baked font.
#[derive(Clone, Debug, PartialEq)]
pub struct Baked {
    /// The TrueType file.
    pub input: PathBuf,
    /// The `.anfont` file written.
    pub output: PathBuf,
    /// Glyphs in the atlas (the fallback included).
    pub glyphs: usize,
    /// Kerning pairs between the baked glyphs.
    pub kerning_pairs: usize,
    /// Atlas pages.
    pub pages: u16,
    /// Side of a page, texels.
    pub page_size: u32,
    /// Size of the `.anfont` file.
    pub bytes: usize,
    /// Time the bake took.
    pub seconds: f64,
}

impl Baked {
    /// The line `fontbake` prints for this font.
    pub fn line(&self) -> String {
        format!(
            "{} -> {}: {} glyphs, {} kerning pairs, {} page(s) of {}^2, {} bytes, {:.2} s",
            self.input.display(),
            self.output.display(),
            self.glyphs,
            self.kerning_pairs,
            self.pages,
            self.page_size,
            self.bytes,
            self.seconds
        )
    }
}

/// Reads the font `input` (at most [`MAX_FONT_BYTES`]).
fn read_font(input: &Path) -> Result<Vec<u8>, CliError> {
    let size = std::fs::metadata(input).map_err(|e| CliError::io(input, e))?.len();
    if size > MAX_FONT_BYTES {
        return Err(CliError::Data(format!(
            "{}: {size} bytes is more than a font may have ({MAX_FONT_BYTES})",
            input.display()
        )));
    }
    std::fs::read(input).map_err(|e| CliError::io(input, e))
}

/// Bakes `input` into `output`.
fn bake_one(input: &Path, output: &Path, icons: bool) -> Result<Baked, CliError> {
    let ttf = read_font(input)?;
    let start = Instant::now();
    let baked = bake_shipped(&ttf, icons).map_err(|e| CliError::Data(format!("{}: {e}", input.display())))?;
    let bytes = baked.to_bytes();
    let seconds = start.elapsed().as_secs_f64();
    std::fs::write(output, &bytes).map_err(|e| CliError::io(output, e))?;
    Ok(Baked {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        glyphs: baked.glyphs.len(),
        kerning_pairs: baked.kerning.len(),
        pages: baked.pages,
        page_size: baked.page_size,
        bytes: bytes.len(),
        seconds,
    })
}

/// Runs `job`, calling `done` after each font is written, and returns what was baked.
///
/// # Errors
/// [`CliError::Io`] for a font that cannot be read or an `.anfont` that cannot be written,
/// [`CliError::Data`] for a font the baker refuses. Fonts baked before the failing one stay
/// written.
pub fn run(job: &FontbakeJob, mut done: impl FnMut(&Baked)) -> Result<Vec<Baked>, CliError> {
    let mut out = Vec::new();
    match job {
        FontbakeJob::One { font, out: output, icons } => {
            let baked = bake_one(font, output, *icons)?;
            done(&baked);
            out.push(baked);
        }
        FontbakeJob::Shipped { dir } => {
            for face in FontFace::ALL {
                let output = dir.join(format!("{}.anfont", face.name()));
                let baked = bake_one(&dir.join(source_file(face)), &output, face == FontFace::Icons)?;
                done(&baked);
                out.push(baked);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use humanart_ui::font::BakedFont;

    use super::*;

    fn job(args: &[&str]) -> Result<FontbakeJob, CliError> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        FontbakeJob::from_args(&Parsed::parse(&FONTBAKE, &args)?)
    }

    fn fonts() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fonts")
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("humanart-cli-fontbake-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ui_03_fontbake_command_lines() {
        assert_eq!(job(&[]).unwrap(), FontbakeJob::Shipped { dir: PathBuf::from(DEFAULT_DIR) });
        assert_eq!(job(&["--dir", "fonts"]).unwrap(), FontbakeJob::Shipped { dir: PathBuf::from("fonts") });
        let one = |icons| FontbakeJob::One { font: PathBuf::from("a.ttf"), out: PathBuf::from("a.anfont"), icons };
        assert_eq!(job(&["--font", "a.ttf", "--out", "a.anfont"]).unwrap(), one(false));
        assert_eq!(job(&["--font", "a.ttf", "--out", "a.anfont", "--kind", "text"]).unwrap(), one(false));
        assert_eq!(job(&["--font=a.ttf", "--out=a.anfont", "--kind=icons"]).unwrap(), one(true));
        for bad in [
            &["--font", "a.ttf"][..],
            &["--out", "a.anfont"],
            &["--kind", "icons"],
            &["--dir", "d", "--font", "a.ttf", "--out", "a.anfont"],
            &["--font", "a.ttf", "--out", "a.anfont", "--kind", "emoji"],
            &["a.ttf"],
        ] {
            assert!(matches!(job(bad), Err(CliError::Usage(_))), "{bad:?}");
        }
    }

    #[test]
    fn ui_03_fontbake_reproduces_a_shipped_face() {
        // [UI-03] The command bakes Inter into a file the UI loads, and the bytes are those of the
        // shipped `data/fonts/regular.anfont`: the shipped atlas is what this baker makes of the
        // shipped source.
        let dir = scratch("one");
        let out = dir.join("regular.anfont");
        let job =
            FontbakeJob::One { font: fonts().join(source_file(FontFace::Regular)), out: out.clone(), icons: false };
        let mut lines = Vec::new();
        let baked = run(&job, |b| lines.push(b.line())).unwrap();
        assert_eq!(baked.len(), 1);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("regular.anfont") && lines[0].contains("glyphs"), "{}", lines[0]);
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(bytes.len(), baked[0].bytes);
        let font = BakedFont::from_bytes(&bytes).unwrap();
        assert_eq!(font.glyphs.len(), baked[0].glyphs);
        assert!(baked[0].glyphs > 300 && baked[0].kerning_pairs > 1000 && baked[0].pages >= 1, "{:?}", baked[0]);
        eprintln!("{}", lines[0]);
        let shipped = std::fs::read(fonts().join("regular.anfont")).unwrap();
        assert!(bytes == shipped, "the baked face differs from the shipped data/fonts/regular.anfont");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ui_03_fontbake_failures_leave_no_file() {
        let dir = scratch("bad");
        // A missing source: nothing is written, the error names the file.
        let shipped = run(&FontbakeJob::Shipped { dir: dir.clone() }, |_| {});
        assert!(
            matches!(&shipped, Err(CliError::Io { path, .. }) if path.ends_with("Inter-Regular.ttf")),
            "{shipped:?}"
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        // Not a font: a data error, and the existing output keeps its bytes.
        let (font, out) = (dir.join("junk.ttf"), dir.join("junk.anfont"));
        std::fs::write(&font, b"this is not a TrueType font").unwrap();
        std::fs::write(&out, b"before").unwrap();
        let result = run(&FontbakeJob::One { font, out: out.clone(), icons: true }, |_| {});
        assert!(matches!(&result, Err(CliError::Data(text)) if text.contains("junk.ttf")), "{result:?}");
        assert_eq!(std::fs::read(&out).unwrap(), b"before");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
