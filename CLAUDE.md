# CLAUDE.md

HumanArt is a native Rust + wgpu studio and runtime library for realistic human characters, edited by a
human and an AI agent through the same command system. `todo.md` is the plan: what is still to do, most
important first (photorealism leads). The original specification was removed on 2026-11-19; its
requirement IDs (`[DEP-03]`, `[AC-WP07-1]`) stay in code comments and tests as labels, and new work
cites the `todo.md` item instead. `PROGRESS.md` says where every work package stands and holds the
evidence; `docs/decisions.md` holds the ADRs.
`docs/reference-conventions.md` describes the reference application (AtroxaBot,
`C:\Users\jerry\Documents\atroxabot-app`, read-only) whose platform plumbing and look we follow.

The product is **HumanArt**. The packages are `humanart-core`, `humanart-math`, ... in `crates/core`,
`crates/math`, ..., the editor is the package and binary `humanart` in `crates/app`, and the CLI is
`humanart-cli` in `crates/cli`. Never use an `an-` crate prefix or the old codename Anthropos in code,
file names or prose; only the existing file extensions (`.anthropos`, `.anchar`, `.antmpl`) keep it
until the owner decides on them.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
sh tools/check-tree.sh      # stubs, unwraps, unsafe, shaders, evidence files
sh tools/check-deps.sh      # cargo tree against docs/dependencies.md
cargo check --target wasm32-unknown-unknown -p humanart-core -p humanart-math -p humanart-io -p humanart-model -p humanart-doc -p humanart-gpu -p humanart-shaders -p humanart-render -p humanart-ui -p humanart-runtime
```

## Hard rules (the check scripts enforce most of them)

- **Dependencies: `wgpu` and `winit`, nothing else.** No serde, bytemuck, thiserror, anyhow, log, rayon,
  image, png, glam, tokio, clap, rfd, arboard or any other crate, not even as a dev-dependency. What is
  needed is written in-house, in the crate whose job it is (`docs/architecture.md`, `[DEP-05]`). No proc macros, no
  build scripts.
- **Errors** are hand-written enums per crate with `Display` + `std::error::Error` (+ `source()`).
  Library code never panics on recoverable conditions: bad files, missing assets, out-of-range
  parameters, network input and GPU loss return `Result`.
- **No `unwrap()` / `expect()` outside tests** (clippy denies them). A true internal invariant may use
  `expect` under `#[allow(clippy::expect_used)]` with a `// INVARIANT: why this cannot fail` comment on
  the line above the call. Indexing that can go out of bounds on outside input is the same bug: check
  first.
- **No stubs:** no `todo!()`, `unimplemented!()`, placeholder `unreachable!()`, functions that return
  fake data, and no `TODO` / `FIXME` / `XXX` comments. If something cannot be done, say so to whoever
  gave you the task instead of leaving a shell.
- **Unsafe:** `#![forbid(unsafe_code)]` everywhere except `humanart-gpu` (POD casts), `humanart-platform` (OS
  shims) and `humanart-core::alloc`. Every `unsafe` block has a `// SAFETY:` comment and a test at its
  boundary.
- **wasm:** `humanart-core`, `humanart-math`, `humanart-io`, `humanart-model`, `humanart-doc`, `humanart-gpu`, `humanart-shaders`, `humanart-render`,
  `humanart-ui` and `humanart-runtime` compile for `wasm32-unknown-unknown`: no `std::fs`, `std::net`,
  `std::thread::spawn`, `std::process`, `std::time::Instant` / `SystemTime` (they panic there) in
  those crates outside `#[cfg(test)]`. Files, clocks and threads come through the traits in
  `humanart_core::platform`, implemented by the native binaries.
- **Parsers survive hostile input** `[ROB-01]`: every length, count, offset and dimension read from a
  file is checked against the remaining input and a documented limit before anything is allocated or
  indexed; no recursion without a depth limit; no loop without guaranteed progress. Every parser has a
  mutation-fuzz test driven by `humanart_core::rng::Rng` with fixed seeds.
- **Performance** `[PERF-R1..R8]`: no heap allocation in steady-state per-frame paths (render, UI,
  deformation dispatch, brush strokes); structure-of-arrays for per-vertex and per-strand data; no
  trait objects in inner loops; pipelines and bind groups are created once and cached; GPU readbacks
  are asynchronous.
- **Shaders** are WGSL files under `shaders/`, never generated from user text; every one is listed in
  the manifest in `humanart-shaders` and validated by naga in tests.
- **Code quality:** every public item has a doc comment, `clippy::pedantic` stays clean (the allowed
  lints are in `Cargo.toml`), non-trivial logic has unit tests in a `#[cfg(test)] mod tests` at the END
  of its file (the check script treats everything after the first `#[cfg(test)]` as test code).
  Binary formats are little-endian, written and read explicitly (`to_le_bytes`), never by casting
  structs. Text files and sources use LF.
- **Honesty:** never invent rubric scores, consent records, provenance entries, benchmark numbers or
  dataset coverage. A claim of "done" comes with the test, benchmark or picture that shows it.

## Conventions

- Right-handed, Y up, metres, the character faces +Z. Matrices are column-major (`Mat4::cols`), a
  point transforms as `M * p`. Quaternions are `(x, y, z, w)`. UV origin is the top-left, V down.
  Depth is reversed-Z, 32-bit float. Colours are linear Rec.709 unless a name says `srgb`.
- Dependency direction (section 3.2): `humanart-core` ← `humanart-math` ← `humanart-io` ← `humanart-model` ← `humanart-doc`;
  `humanart-gpu` ← `humanart-shaders` ← `humanart-render`; `humanart-ui` uses `humanart-gpu` + `humanart-shaders` and knows nothing of
  characters; `humanart-model` and `humanart-doc` use no GPU code; `humanart-runtime` uses neither `humanart-doc` nor
  `humanart-ui`.
- Names in the registry, landmarks and regions are stable ids: never reuse or rename one.
- The owner tests the GUI himself. Never drive the app on his screen; look at a visual change through
  the offscreen renders (`humanart-cli render`, the snapshot tests).
