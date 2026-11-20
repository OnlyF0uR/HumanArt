//! The in-house WGSL preprocessor: `#include` composition and `#ifdef` on capability flags.
//!
//! WGSL has no module system, so shaders are composed from files at load or test time. A directive
//! is a line whose first non-blank character is `#` (WGSL itself never starts a line with `#`):
//!
//! - `#include "lib/math.wgsl"` inserts that file, named by its path under `shaders/`. A file is
//!   inserted at most once per composition (WGSL forbids redeclaring a function), so diamond
//!   includes are fine; a file that includes itself through any chain is an
//!   [`ShaderError::IncludeCycle`].
//! - `#ifdef FLAG`, `#ifndef FLAG`, `#else`, `#endif` keep or drop lines depending on the
//!   capability flags the caller passes (see [`crate::manifest::capability_flag`]).
//!
//! Anything else after `#` is an error, so a typo never silently drops code. Directive lines and
//! dropped lines produce no output; the [`Composed`] result keeps a line map so that a compiler
//! error in the composed text is reported against the original file and line.
//!
//! Limits (`[ROB-01]`; the input is embedded and trusted, but the preprocessor still terminates
//! with an error on any input): includes nest at most [`MAX_INCLUDE_DEPTH`] deep, conditionals
//! [`MAX_CONDITIONAL_DEPTH`] deep, and the output is at most [`MAX_OUTPUT_BYTES`]. Every line is
//! visited once per included file and every file at most once, so composition always ends.

use crate::error::{Location, ShaderError};

/// The deepest chain of nested `#include`s (the root file is depth 0).
pub const MAX_INCLUDE_DEPTH: usize = 16;
/// The deepest nesting of `#ifdef`/`#ifndef` within one file.
pub const MAX_CONDITIONAL_DEPTH: usize = 16;
/// The largest composed source, in bytes.
pub const MAX_OUTPUT_BYTES: usize = 4 << 20;
/// The longest include path or flag name accepted.
const MAX_NAME_LEN: usize = 128;

/// A run of composed lines that come from consecutive lines of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
    /// The first composed line of the run (1-based).
    output_line: u32,
    /// Index into [`Composed::files`].
    file: u32,
    /// The source line of `output_line` (1-based).
    source_line: u32,
}

/// A composed shader: the WGSL text and the map from its lines back to the source files.
#[derive(Debug, Clone)]
pub struct Composed {
    /// The composed WGSL.
    source: String,
    /// Every file that contributed, in order of first inclusion; the root first.
    files: Vec<String>,
    /// Line runs, sorted by `output_line`.
    segments: Vec<Segment>,
    /// The number of composed lines.
    lines: u32,
}

impl Composed {
    /// The composed WGSL text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Consumes the result and returns the text.
    pub fn into_source(self) -> String {
        self.source
    }

    /// The files that contributed, the root first, each once.
    pub fn files(&self) -> impl ExactSizeIterator<Item = &str> {
        self.files.iter().map(String::as_str)
    }

    /// The number of lines of the composed text.
    pub fn line_count(&self) -> u32 {
        self.lines
    }

    /// The file and line that composed line `line` (1-based) came from; `None` outside the text.
    pub fn locate(&self, line: u32) -> Option<Location> {
        if line == 0 || line > self.lines {
            return None;
        }
        let index = self.segments.partition_point(|segment| segment.output_line <= line).checked_sub(1)?;
        let segment = self.segments.get(index)?;
        let file = self.files.get(segment.file as usize)?;
        Some(Location { file: file.clone(), line: segment.source_line + (line - segment.output_line) })
    }
}

/// One open `#ifdef`/`#ifndef`.
#[derive(Debug, Clone, Copy)]
struct Conditional {
    /// Whether the enclosing code is kept.
    parent_active: bool,
    /// Whether the condition held.
    condition: bool,
    /// Whether `#else` was seen.
    in_else: bool,
    /// The line of the `#ifdef`, for the "never closed" error.
    line: u32,
}

impl Conditional {
    /// Whether lines under this conditional are kept.
    fn active(self) -> bool {
        self.parent_active && (self.condition != self.in_else)
    }
}

/// True for `[A-Za-z_][A-Za-z0-9_]*` of at most [`MAX_NAME_LEN`] bytes.
pub fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= MAX_NAME_LEN
        && chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// True for a path under `shaders/`: `/`-separated components of `[A-Za-z0-9_.-]`, none empty, `.`
/// or `..`, ending in `.wgsl`.
pub fn is_shader_path(path: &str) -> bool {
    path.len() <= MAX_NAME_LEN
        && path.strip_suffix(".wgsl").is_some_and(|stem| !stem.is_empty() && !stem.ends_with('/'))
        && path.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        })
}

/// The composition in progress.
struct State<'s, 'f, R: Fn(&str) -> Option<&'s str>> {
    /// Finds a file's text by path.
    resolve: R,
    /// The capability flags that are set.
    flags: &'f [&'f str],
    /// The output.
    out: String,
    /// Contributing files.
    files: Vec<String>,
    /// Line runs.
    segments: Vec<Segment>,
    /// Lines written.
    lines: u32,
    /// The next `(file, line)` that would continue the last segment.
    continues: Option<(u32, u32)>,
    /// Files currently being expanded, outermost first.
    stack: Vec<String>,
    /// The root path, for the size error.
    root: String,
}

impl<'s, R: Fn(&str) -> Option<&'s str>> State<'s, '_, R> {
    /// Appends one kept line of `file` (index) at `line`.
    fn emit(&mut self, file: u32, line: u32, text: &str) -> Result<(), ShaderError> {
        if self.out.len() + text.len() + 1 > MAX_OUTPUT_BYTES {
            return Err(ShaderError::OutputTooLarge { root: self.root.clone() });
        }
        self.lines += 1;
        if self.continues != Some((file, line)) {
            self.segments.push(Segment { output_line: self.lines, file, source_line: line });
        }
        self.continues = Some((file, line + 1));
        self.out.push_str(text);
        self.out.push('\n');
        Ok(())
    }

    /// Expands `path` (included from `from`, at nesting `depth`).
    fn expand(&mut self, path: &str, depth: usize, from: Option<&Location>) -> Result<(), ShaderError> {
        if let Some(position) = self.stack.iter().position(|open| open == path) {
            let mut chain: Vec<String> = self.stack.get(position..).unwrap_or_default().to_vec();
            chain.push(path.to_owned());
            return Err(ShaderError::IncludeCycle { chain });
        }
        if self.files.iter().any(|done| done == path) {
            return Ok(());
        }
        if depth > MAX_INCLUDE_DEPTH {
            let at = from.cloned().unwrap_or_else(|| Location { file: path.to_owned(), line: 0 });
            return Err(ShaderError::IncludeTooDeep { at });
        }
        let Some(text) = (self.resolve)(path) else {
            return Err(ShaderError::UnknownFile { path: path.to_owned(), included_from: from.cloned() });
        };
        let file =
            u32::try_from(self.files.len()).map_err(|_| ShaderError::OutputTooLarge { root: self.root.clone() })?;
        self.files.push(path.to_owned());
        self.stack.push(path.to_owned());
        let mut conditionals: Vec<Conditional> = Vec::new();
        let mut line_number: u32 = 0;
        for line in text.lines() {
            line_number = line_number.saturating_add(1);
            let active = conditionals.last().is_none_or(|top| top.active());
            let trimmed = line.trim_start();
            let Some(directive) = trimmed.strip_prefix('#') else {
                if active {
                    self.emit(file, line_number, line)?;
                }
                continue;
            };
            let at = || Location { file: path.to_owned(), line: line_number };
            let (word, rest) = match directive.find(char::is_whitespace) {
                Some(split) => (directive.get(..split).unwrap_or(""), directive.get(split..).unwrap_or("").trim()),
                None => (directive, ""),
            };
            match word {
                "include" => {
                    let target = rest
                        .strip_prefix('"')
                        .and_then(|r| r.strip_suffix('"'))
                        .filter(|target| is_shader_path(target))
                        .ok_or_else(|| ShaderError::BadDirective {
                            at: at(),
                            message: "#include needs one quoted path under shaders/ ending in .wgsl",
                        })?;
                    if active {
                        self.expand(target, depth + 1, Some(&at()))?;
                    }
                }
                "ifdef" | "ifndef" => {
                    if !is_identifier(rest) {
                        return Err(ShaderError::BadDirective { at: at(), message: "#ifdef needs one flag name" });
                    }
                    if conditionals.len() >= MAX_CONDITIONAL_DEPTH {
                        return Err(ShaderError::ConditionalTooDeep { at: at() });
                    }
                    let set = self.flags.contains(&rest);
                    let condition = if word == "ifdef" { set } else { !set };
                    conditionals.push(Conditional {
                        parent_active: active,
                        condition,
                        in_else: false,
                        line: line_number,
                    });
                }
                "else" | "endif" => {
                    if !rest.is_empty() {
                        return Err(ShaderError::BadDirective { at: at(), message: "text after #else or #endif" });
                    }
                    let Some(top) = conditionals.last_mut() else {
                        return Err(ShaderError::Unbalanced { at: at(), message: "#else or #endif without #ifdef" });
                    };
                    if word == "endif" {
                        conditionals.pop();
                    } else if top.in_else {
                        return Err(ShaderError::Unbalanced { at: at(), message: "a second #else" });
                    } else {
                        top.in_else = true;
                    }
                }
                other => {
                    let directive: String = other.chars().take(MAX_NAME_LEN).collect();
                    return Err(ShaderError::UnknownDirective { at: at(), directive });
                }
            }
            // The next kept line does not continue the run before the directive.
            self.continues = None;
        }
        if let Some(open) = conditionals.last() {
            let at = Location { file: path.to_owned(), line: open.line };
            return Err(ShaderError::Unbalanced { at, message: "#ifdef never closed by #endif" });
        }
        self.stack.pop();
        self.continues = None;
        Ok(())
    }
}

/// Composes `root` with the capability `flags` set, reading files through `resolve` (path under
/// `shaders/` to text).
pub fn compose_with<'s>(
    root: &str,
    flags: &[&str],
    resolve: impl Fn(&str) -> Option<&'s str>,
) -> Result<Composed, ShaderError> {
    if let Some(flag) = flags.iter().find(|flag| !is_identifier(flag)) {
        return Err(ShaderError::BadFlag { flag: flag.chars().take(MAX_NAME_LEN).collect() });
    }
    let mut state = State {
        resolve,
        flags,
        out: String::new(),
        files: Vec::new(),
        segments: Vec::new(),
        lines: 0,
        continues: None,
        stack: Vec::new(),
        root: root.to_owned(),
    };
    state.expand(root, 0, None)?;
    Ok(Composed { source: state.out, files: state.files, segments: state.segments, lines: state.lines })
}

/// Composes the embedded shader `root` (a path under `shaders/`) with the capability `flags` set.
pub fn compose(root: &str, flags: &[&str]) -> Result<Composed, ShaderError> {
    compose_with(root, flags, crate::manifest::source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(table: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<&'static str> {
        move |path| table.iter().find(|(name, _)| *name == path).map(|(_, text)| *text)
    }

    #[test]
    fn includes_are_inserted_once_and_lines_map_back() {
        let table: &[(&str, &str)] = &[
            ("main.wgsl", "#include \"lib/a.wgsl\"\n#include \"lib/b.wgsl\"\nfn main_fn() {}\n"),
            ("lib/a.wgsl", "// a\nfn a() {}\n"),
            ("lib/b.wgsl", "#include \"lib/a.wgsl\"\nfn b() { a(); }\n"),
        ];
        let composed = compose_with("main.wgsl", &[], files(table)).unwrap();
        assert_eq!(composed.source(), "// a\nfn a() {}\nfn b() { a(); }\nfn main_fn() {}\n");
        assert_eq!(composed.files().collect::<Vec<_>>(), ["main.wgsl", "lib/a.wgsl", "lib/b.wgsl"]);
        assert_eq!(composed.line_count(), 4);
        let at = |line| composed.locate(line).map(|l| (l.file, l.line));
        assert_eq!(at(1), Some(("lib/a.wgsl".to_owned(), 1)));
        assert_eq!(at(2), Some(("lib/a.wgsl".to_owned(), 2)));
        assert_eq!(at(3), Some(("lib/b.wgsl".to_owned(), 2)));
        assert_eq!(at(4), Some(("main.wgsl".to_owned(), 3)));
        assert_eq!(at(0), None);
        assert_eq!(at(5), None);
    }

    #[test]
    fn ifdef_follows_the_flags() {
        let table: &[(&str, &str)] = &[(
            "m.wgsl",
            "a\n#ifdef SHADER_F16\nf16\n#ifndef SUBGROUPS\nnosub\n#else\nsub\n#endif\n#else\nf32\n#endif\nz\n",
        )];
        let with = |flags: &'static [&'static str]| compose_with("m.wgsl", flags, files(table)).unwrap();
        assert_eq!(with(&[]).source(), "a\nf32\nz\n");
        assert_eq!(with(&["SHADER_F16"]).source(), "a\nf16\nnosub\nz\n");
        assert_eq!(with(&["SHADER_F16", "SUBGROUPS"]).source(), "a\nf16\nsub\nz\n");
        // Line mapping skips the dropped lines.
        let composed = with(&["SHADER_F16", "SUBGROUPS"]);
        assert_eq!(composed.locate(3).map(|l| l.line), Some(7));
        assert_eq!(composed.locate(4).map(|l| l.line), Some(12));
    }

    #[test]
    fn includes_inside_dropped_branches_are_not_read() {
        let table: &[(&str, &str)] = &[("m.wgsl", "#ifdef X\n#include \"missing.wgsl\"\n#endif\nok\n")];
        assert_eq!(compose_with("m.wgsl", &[], files(table)).unwrap().source(), "ok\n");
        let error = compose_with("m.wgsl", &["X"], files(table)).unwrap_err();
        assert!(matches!(error, ShaderError::UnknownFile { ref path, included_from: Some(ref at) }
            if path == "missing.wgsl" && at.line == 2));
    }

    #[test]
    fn cycles_are_reported_with_their_chain() {
        let table: &[(&str, &str)] = &[
            ("a.wgsl", "#include \"b.wgsl\"\n"),
            ("b.wgsl", "#include \"c.wgsl\"\n"),
            ("c.wgsl", "#include \"b.wgsl\"\n"),
            ("self.wgsl", "#include \"self.wgsl\"\n"),
        ];
        let error = compose_with("a.wgsl", &[], files(table)).unwrap_err();
        let ShaderError::IncludeCycle { chain } = error else { panic!("expected a cycle, got {error}") };
        assert_eq!(chain, ["b.wgsl", "c.wgsl", "b.wgsl"]);
        assert!(matches!(compose_with("self.wgsl", &[], files(table)), Err(ShaderError::IncludeCycle { .. })));
    }

    #[test]
    fn include_depth_is_limited() {
        // d0 includes d1 includes ... d20: deeper than the limit.
        static CHAIN: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
        let chain = CHAIN.get_or_init(|| {
            (0..=20)
                .map(|i| (format!("d{i}.wgsl"), format!("#include \"d{}.wgsl\"\nfn f{i}() {{}}\n", i + 1)))
                .collect()
        });
        let resolve = |path: &str| chain.iter().find(|(name, _)| name == path).map(|(_, text)| text.as_str());
        let error = compose_with("d0.wgsl", &[], resolve).unwrap_err();
        let ShaderError::IncludeTooDeep { at } = error else { panic!("expected the depth limit, got {error}") };
        assert_eq!(at.file, format!("d{MAX_INCLUDE_DEPTH}.wgsl"));
        // Exactly at the limit is fine.
        let shallow = |path: &str| {
            if path == format!("d{MAX_INCLUDE_DEPTH}.wgsl") { Some("fn leaf() {}\n") } else { resolve(path) }
        };
        assert!(compose_with("d0.wgsl", &[], shallow).is_ok());
    }

    #[test]
    fn malformed_directives_are_errors() {
        type Check = fn(&ShaderError) -> bool;
        let cases: &[(&str, Check)] = &[
            ("#include lib/a.wgsl\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#include \"../a.wgsl\"\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#include \"/abs.wgsl\"\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#include \"a.txt\"\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#define X\n", |e| matches!(e, ShaderError::UnknownDirective { .. })),
            ("#ifdef\n#endif\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#ifdef 9X\n#endif\n", |e| matches!(e, ShaderError::BadDirective { .. })),
            ("#endif\n", |e| matches!(e, ShaderError::Unbalanced { .. })),
            ("#ifdef X\n#else\n#else\n#endif\n", |e| matches!(e, ShaderError::Unbalanced { .. })),
            ("#ifdef X\n", |e| matches!(e, ShaderError::Unbalanced { at, .. } if at.line == 1)),
            ("#endif junk\n", |e| matches!(e, ShaderError::BadDirective { .. })),
        ];
        for (text, check) in cases {
            let error = compose_with("m.wgsl", &[], |_| Some(*text)).unwrap_err();
            assert!(check(&error), "{text:?} gave {error}");
        }
        assert!(matches!(compose_with("m.wgsl", &["no-dash"], |_| Some("")), Err(ShaderError::BadFlag { .. })));
    }

    #[test]
    fn conditional_depth_is_limited() {
        let deep = "#ifdef X\n".repeat(MAX_CONDITIONAL_DEPTH + 1) + &"#endif\n".repeat(MAX_CONDITIONAL_DEPTH + 1);
        let error = compose_with("m.wgsl", &[], |_| Some(deep.as_str())).unwrap_err();
        assert!(
            matches!(error, ShaderError::ConditionalTooDeep { ref at } if at.line as usize == MAX_CONDITIONAL_DEPTH + 1)
        );
        let ok = "#ifdef X\n".repeat(MAX_CONDITIONAL_DEPTH) + &"#endif\n".repeat(MAX_CONDITIONAL_DEPTH);
        assert!(compose_with("m.wgsl", &[], |_| Some(ok.as_str())).is_ok());
    }

    #[test]
    fn output_size_is_limited() {
        let big = "x".repeat(1024) + "\n";
        let huge = big.repeat(MAX_OUTPUT_BYTES / big.len() + 1);
        assert!(matches!(
            compose_with("m.wgsl", &[], |_| Some(huge.as_str())),
            Err(ShaderError::OutputTooLarge { .. })
        ));
    }

    #[test]
    fn crlf_and_indented_directives_work() {
        let text = "a\r\n  #ifdef X\r\nb\r\n  #endif\r\nc\r\n";
        assert_eq!(compose_with("m.wgsl", &[], |_| Some(text)).unwrap().source(), "a\nc\n");
    }

    #[test]
    fn paths_and_identifiers() {
        assert!(is_shader_path("lib/math.wgsl"));
        assert!(is_shader_path("post/tone-map_2.wgsl"));
        for bad in ["", "math", "lib//a.wgsl", "./a.wgsl", "a/../b.wgsl", "lib/a b.wgsl", "lib\\a.wgsl", ".wgsl/"] {
            assert!(!is_shader_path(bad), "{bad}");
        }
        assert!(is_identifier("SHADER_F16") && is_identifier("_x1"));
        assert!(!is_identifier("") && !is_identifier("1X") && !is_identifier("A-B"));
    }

    /// `[ROB-01]` mutation fuzz: random edits of directive-heavy sources never panic or hang; every
    /// result is either an error or a composition whose line map covers every line.
    #[test]
    fn rob_01_mutation_fuzz_never_panics() {
        let corpus: Vec<Vec<u8>> = [
            "#include \"b.wgsl\"\n#ifdef X\nfn a() {}\n#else\nfn b() {}\n#endif\n",
            "#ifndef Y\n#include \"a.wgsl\"\n#endif\nfn c() {}\n",
            "#include \"c.wgsl\"\n#include \"a.wgsl\"\n",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        humanart_core::fuzz::run(
            "preprocess",
            &corpus,
            4000,
            0x5eed_0005,
            std::time::Duration::from_secs(60),
            |bytes| {
                let mutated = String::from_utf8_lossy(bytes).into_owned();
                let others = ["#include \"a.wgsl\"\nfn b() {}\n", "#ifdef X\n#include \"b.wgsl\"\n#endif\n"];
                let resolve = |path: &str| match path {
                    "a.wgsl" => Some(mutated.as_str()),
                    "b.wgsl" => Some(others[0]),
                    "c.wgsl" => Some(others[1]),
                    _ => None,
                };
                for flags in [&[][..], &["X"][..]] {
                    if let Ok(composed) = compose_with("a.wgsl", flags, resolve) {
                        for line in 1..=composed.line_count() {
                            assert!(composed.locate(line).is_some());
                        }
                    }
                }
            },
        );
    }
}
