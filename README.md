# HumanArt

HumanArt is a studio and runtime library for realistic human characters, written in Rust on
[wgpu](https://wgpu.rs) and [winit](https://github.com/rust-windowing/winit). A person and an AI
agent edit the same character through one command system, so every change has the same history,
undo and review, whoever made it.

**Status: early and under heavy development.** The engineering base is in place and tested (image
and mesh codecs, GPU layer, render graph, UI toolkit, skin model calibrated against measured skin
colour, subsurface shading, eyes, mouth interior, strand hair, LODs). The character itself does not
look photorealistic yet; `todo.md` lists what is still to do, most important first, and
`PROGRESS.md` holds the evidence for what is done.

## Build and test

Rust 1.99 (pinned in `rust-toolchain.toml`) and a GPU that wgpu supports. So far it has only been
built and tested on Windows with an NVIDIA GPU; macOS and Linux are untested.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p humanart-cli -- render --help   # offscreen renders
cargo run -p humanart                        # the editor (not yet tried on screen)
```

## Layout

- `crates/`: one crate per job (`humanart-core`, `-math`, `-io`, `-model`, `-doc`, `-gpu`, `-shaders`,
  `-render`, `-ui`, `-runtime`, ...); the editor is `crates/app`, the command line `crates/cli`.
  `docs/architecture.md` explains how they depend on each other.
- `shaders/`: every WGSL shader, validated in the tests.
- `data/`: fonts, template parameters, anthropometry and skin reference data.
- `docs/`: architecture, file formats, decisions (`decisions.md`), data provenance.

The only third-party crates are `wgpu` and `winit`; everything else is written in-house
(`docs/dependencies.md`). `CLAUDE.md` holds the rules every change follows.

## Licence

HumanArt is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you state otherwise, any contribution you submit for inclusion is licensed
as above, without any additional terms.

Data and fonts keep their own licences (SIL Open Font License, ISC, CC BY 4.0, CC BY-SA 4.0 and
others); `DATA-ATTRIBUTION.md` gives the credits those licences require and `docs/data-provenance.md` lists every file with its source and licence.
