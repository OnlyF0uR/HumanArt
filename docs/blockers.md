# Blockers

What is blocked, and exactly what unblocks it (spec 0.1.4). Each entry names the default that was
taken meanwhile and the ADR that records it.

## B-001: No captured scan data exists

- **Blocks.** Fitted identity, expression and appearance models (`[ID-01]`, `[STAT-01..11]`,
  `[APR-01..06]`), the coverage gate (`[DATA-13]`), and every realism judgement that depends on real
  anatomy.
- **Needs.** Scans captured under `docs/capture-protocol.md` with signed releases (`[DATA-01]`), or a
  dataset whose licence permits building a shipped statistical model (`[PUB-01]`).
- **Meanwhile.** The no-data model (`[ID-07]`): generated template, authored axes (ADR-006). The capture
  and model-building tools are built and tested on synthetic data.
- **State (2026-11-19, round 5).** Unchanged. The identity agent's 62 authored components
  (`humanart_model::identity`, ADR-048) are model choices, not statistics. The ANSUR II summary
  (`data/anthropometry/ansur2.json`, ADR-046) gives per-sex measurement statistics and correlations of
  US Army soldiers, aggregates only: it can calibrate axis ranges and a measurement prior, but it
  holds no shapes and does not lift this blocker.
- **State (round 6).** Unchanged. Identity round two calibrates 14 axes so one unit moves an ANSUR II
  measurement by one SD (ADR-051) and sets the joint-regressor rows from the skin landmark rings of
  each authored component; the shapes, the template's skin weights (ADR-049) and its surface relief
  remain authored, not fitted.

## B-002: Human review sign-offs

- **Blocks.** The review rubric scores (15.4, `[AC-WP14-2]`, `[AC-WP16-1]`, `[AC-WP18-1]`,
  `[AC-WP19-2]`, `[AC-WP35-2]`) and golden-image sign-off (15.3). An agent must not fabricate them
  (17.2 rule 7).
- **Needs.** A human reviewer scoring the rendered scenes in `docs/review-rubric.md`.
- **State (2026-11-19, round 4).** Waiting for a reviewer: the WP-14 sidelight and backlight
  stills of three skin types plus a no-skin-shading control
  (`target/agents/render/shots/wp14_{light,medium,dark,medium-no-skin-shading}_{sidelight,backlight}.png`,
  written by `tests/skin.rs::ac_wp14_2_sidelight_and_backlight_stills`; no score given by the agent),
  the expression previews (`target/agents/expression/previews/`) and the hair contact sheets
  (`target/agents/hair/*.png`). These are outside the repository and are rewritten by the tests.
  The lead's own look at `previews/presets-0.png`: the expressions read weakly, smiles are barely
  visible and the eyes have no iris; that is not a rubric score. Review scene 8 (`[AC-WP19-2]`)
  does not exist yet.
- **State (round 5).** The pigment model changed (ADR-047: packaged eumelanin, melanin range 0..1),
  so the WP-14 stills listed above show the old, too-yellow skin colours and must be rendered again
  before review (the skin-calib agent expects darker, less yellow dark skin). New material for the
  owner's eye, no score given: the identity agent's reference comparison
  (`target/agents/identity/previews/`; the agent's own reading in its STATUS: the bust stays much
  smaller and conical, buttock projection about half the reference's) and the skin-finish zone mask
  previews (`target/agents/skin-finish/previews/zones_{front,back,face}.png`).
- **State (round 6).** The pigment model changed again (ADR-053, blood packaging), so the WP-14
  stills still have to be rendered again before review. New material, no score given by any agent:
  the surfaced-skin stills (`target/agents/render/shots/surface_{head_front,head_macro_cheek,body_front}_{plain,surfaced}.png`,
  skin-render; the agent's reading: zone colours and skin lines show in the macro view, the look is
  still smooth and CG-like at portrait distance), the CLI renders
  (`target/agents/skin-wire/shots/cli_*_{plain,surface}.png`; lip colour now only on the
  vermilion), template-gen's previews (`target/agents/template-gen/previews/`: relief, navel,
  nipples, hands, open mouth, arms lowered) and identity's round-two reference comparison
  (`target/agents/identity/previews/reference-round2.png`). These files are rewritten by later runs.

## B-003: Only one platform and one GPU are available to the implementer

- **Blocks.** Building and running on macOS and Linux (`[PLAT-02]`, `[AC-WP01-2]`, `[AC-WP31-1]`,
  `[AC-WP37-1]`), and benchmark baselines for tiers other than the one this machine is
  (`[BENCH-02]`).
- **Needs.** A macOS and a Linux machine (or CI runners), and a Tier L and a Tier M adapter.
- **Meanwhile.** Windows x86_64 (MSVC) is built and tested; the wasm32 build of the library crates is
  checked; platform shims for the other systems are written against their documented APIs and marked
  unverified in `PROGRESS.md`.
- **State (2026-11-19).** `humanart-platform` passes clippy `--all-targets -D warnings` for
  `x86_64-unknown-linux-gnu` and `aarch64-apple-darwin` (type-check only: never linked or run); on
  those systems the clipboard is in-process only and file dialogs report `Unavailable` (ADR-018). The
  manual checklist `docs/platform-checklist.md` has not been run on any OS. Every measured number so
  far comes from one machine (Intel Core i7-10700K, NVIDIA GeForce RTX 4060, Windows 11); whether it
  stands for Tier M is for the owner to say, so `[AC-WP03-3]` stays unconfirmed for Tier M. The UI golden screenshots (`crates/ui/tests/golden/`) and the
  BC7/BC6H hardware-decode parity tests were made on this RTX 4060 only; other GPUs may need
  re-blessing after a visual check.
  Round 3 adds more single-GPU evidence: the widget goldens (`controls`, `popups`, `editors`, `views`,
  `dock`, `bench_scene`, `text_area`), the GPU/CPU parity tests of the deformer, skin compositor
  and bakers, and the render acceptance tests, all on this RTX 4060 (Vulkan, headless) only.
  Round 4 adds the skin SSS furnace, kernel and 1440p timing tests, the groom GPU/CPU parity and
  latency tests, the shell's offscreen shots and the BC7 constant-colour tests, again on this RTX
  4060 only. The test-binary deadlock fixed by ADR-040 was found inside this NVIDIA driver
  (`nvoglv64.dll`, driver 621.01); other drivers were not tried.
  Round 5 adds the wrinkle activation and blend GPU/CPU parity tests and the GPU-to-pool page copy
  test of the skin system (skin-finish agent), again on this RTX 4060 only.
  Round 6 adds the surfaced-skin tests and timing (skin-render), the GPU timestamp timings of the
  texture encoders (perf-gaps) and the CLI's GPU render of the generated template (skin-wire),
  again on this RTX 4060 only. The JPEG decode now meets 100 MP/s on PC-1 (D-002 closed), which
  still says nothing about Tier M until the owner rules on PC-1.

## B-004: Munsell hue and chroma need the 1943 renotation data

- **Blocks.** Munsell hue and chroma readouts in the colour picker (`[SKN-13]`: "the closest
  Munsell/Lab readout"). Munsell Value is done (ASTM D1535, `crates/math/src/color/munsell.rs`).
- **Needs.** The Munsell renotation table (Newhall, Nickerson & Judd 1943; the RIT Munsell Color
  Science Laboratory `real.dat`, xyY under Illuminant C) in the repository with a provenance row and a
  licence the owner accepts. It must not be typed in from memory or invented.
- **Meanwhile.** The picker can show Lab, LCh and Munsell Value (`N 5.2/`, `V 5.2`) only.

## B-005: No PIZ-compressed EXR written by the reference OpenEXR library

- **Blocks.** Independent evidence that the PIZ decoder reads files from other tools (`[IO-03]`).
- **Needs.** One or more small PIZ EXR files written by OpenEXR (or a tool built on it) with known
  pixel values and a licence that allows them as fixtures (generated in-house or CC0). ffmpeg, the
  only independent EXR tool on this PC, cannot write PIZ.
- **Meanwhile.** PIZ is verified by round trip (both wavelet paths, long Huffman codes) and by
  ffmpeg's decoder reading all of our PIZ files to our samples (ADR-011).

## B-006: No skin reference swatch table

- **Data resolved 2026-11-19 (round 5, ADR-047); two owner questions open.**
  `data/test-scenes/skin-reference.json` holds aggregates of ISSA (Lu Y. et al. 2025, Scientific Data
  12, 487, doi:10.1038/s41597-025-04857-5; data doi:10.6084/m9.figshare.28228571.v4; CC BY 4.0):
  15 256 specular-included spectrophotometer records of 2 107 subjects in 8 self-reported groups,
  binned by ITA category, L* band and body site, generated by
  `tools/skin-reference/gen_skin_reference.py` (the lead's rebuild was byte-identical).
  `[AC-WP13-1]` is now tested by `pigment::tests::ac_wp13_1_reproduces_the_issa_reference_swatches`
  and `ac_wp13_1_fitted_spectra_follow_the_measured_spectra` (humanart-model; the lead reran the 62
  `skin_appearance` tests in release: pass). **Open for the owner:** (1) ISSA has no Fitzpatrick
  phototypes; the Fitzpatrick I-VI coverage is read as the six ITA categories (Chardon 1991 /
  Del Bino 2006). (2) The spec names no ΔE tolerance; the tests use ΔE00 <= 1.0 on bin means,
  <= 2.5 on the decile-edge swatches and spectral RMS < 0.035 per ITA bin, and the last two were set
  by the agent after seeing the results. Until the owner accepts both, `[AC-WP13-1]` counts as met
  against agent-chosen terms, not settled. The text below is the blocker as it stood.
  **Round 6:** with blood packaging (ADR-053) the RMS tolerance was tightened from 0.035 to 0.012
  (worst bin 0.0082); the worst bin-mean ΔE00 rose from 0.60 to 0.81 (tolerance 1.0) and the worst
  decile edge fell from 2.14 to 1.86 (tolerance 2.5). The tightened RMS bound was also set after
  seeing the results, so question (2) now covers it.

- **Blocks.** Validation of skin colour against Fitzpatrick I-VI (`[SKN-10]`, `[AC-WP13-1]`, review
  scene 9) and the skin-tone sanity region built from it.
- **Needs.** `data/test-scenes/skin-reference.json`: measured skin colours covering Fitzpatrick I-VI
  with their source, measurement conditions and licence. `data/test-scenes/` is empty. The
  melanin/haemoglobin pigment model (WP-13) also needs absorption data with provenance.
- **Meanwhile.** `humanart_math::color::skin` holds no swatch data; it takes the table from the
  caller (nearest swatch by CIEDE2000, ITA categories, `SkinRegion::enclosing`), so the data can be
  dropped in without code changes.
- **State (2026-11-19, WP-13).** The pigment model now exists (`docs/shader-notes.md` "Skin pigment
  model") with absorption and colour-matching data whose provenance is recorded
  (`docs/data-provenance.md`, WP-13 section; licence: ADR-038). `[AC-WP13-1]` stays blocked:
  `PigmentModel::fit_lab` recovers synthetic Lab targets within ΔE2000 < 0.5, and the tests use only
  the ITA category limits; `data/test-scenes/` is still empty. The colour picker of `humanart-ui`
  takes reference swatches from its caller; its tests use neutral made-up swatches labelled as such.

## B-007: No Khronos glTF Validator and no KTX-Software on this PC

- **Blocks.** Independent evidence that HumanArt's glTF/GLB exports (`[EXP-P1]`, `[EXP-P2]`,
  `[IO-04]`) and KTX2 files (`[IO-10]`) are valid for other tools. The in-house
  `humanart_io::gltf::validate` (`crates/io/src/gltf/validate.rs`) covers its own rule list, not the
  Khronos validator's full set (it does not check that skin weights sum to 1 or that every
  `JOINTS_n` has a matching `WEIGHTS_n`); the KTX2 DFD `sampleLower` / `sampleUpper` words of block
  formats are unchecked (ADR-023).
- **Needs.** The Khronos glTF-Validator and KTX-Software's `ktx validate` installed as dev-time tools
  (`[DEP-04]`), or the owner running them on files the tests write.
- **Meanwhile.** Fixtures from an independent Python writer (`crates/io/tests/fixtures/gltf/`), a
  hand-built KTX2 file the writer reproduces byte for byte, round trips and the in-house validator.

## B-008: Licence decision for the owner: embedding a CC BY-SA 4.0 CIE table subset

- **Resolved 2026-11-19 (owner): ADR-038.** The subset stays under CC BY-SA 4.0 with attribution;
  the licence note is in the file. The text below is the question as it stood.

- **Blocks.** A clean licence position for `crates/model/src/skin_appearance/spectra_table.rs`
  (`[LIC-01]`), which embeds the CIE 1931 2° colour-matching functions and illuminant D65 at 5 nm
  (81 rows each). The CIE publishes both datasets under CC BY-SA 4.0 (DOIs in
  `docs/data-provenance.md`). Whether share-alike reaches the source file (or the binaries built
  from it) when a subset is embedded has not been decided; an agent must not decide it.
- **Needs.** The owner's decision: accept CC BY-SA 4.0 for that file (with attribution, and the
  file marked as such), or replace the table with values from a source under a licence the owner
  accepts, or compute the CMFs from a published analytic fit instead of tabulated data.
- **Related.** The haemoglobin (Prahl, OMLC) and β-carotene (PhotochemCAD via Prahl) tables in the
  same file state no licence; they are attributed only. The generator script and the downloads are
  in `target/agents/skin/`, outside the repository (the script now lives in `tools/spectra/`).
- **Meanwhile.** The table is used as generated (values unchanged) and attributed in its module
  doc and in `docs/data-provenance.md`.

## B-009: The windowed editor has never been launched

- **Blocks.** Evidence that the editor works in a real window: the winit event loop, surface
  configuration and backend choice, native clipboard paste, IME, native folder dialogs, reopening
  the last project on start (`[APP-01]`, `[APP-07]`, the windowed half of `[AC-WP21-1]` and
  `[AC-WP21-3]`). Agents may not open a window on the owner's screen.
- **Needs.** The owner running `cargo run -p humanart` (environment: `HUMANART_DATA`, default
  `./data`; `HUMANART_SETTINGS`, default `%APPDATA%/HumanArt`) and reporting what fails.
- **Meanwhile.** The shell is tested headless and offscreen (`crates/app` tests, 18 pass in the
  lead's rerun, 19 at the perf-gaps and skin-wire agents' runs in round 6; shots in `target/agents/app-shell/shots/` and `target/agents/fixes/shots/`); the
  windowed path (`crates/app/src/main.rs`, `window.rs`) has 2 unit tests for event translation and
  surface configuration.
