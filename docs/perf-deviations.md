# Performance deviations

Every target of section 14.2 that a measurement did not meet, with the measured value, the adapter it
was measured on, the reason and the plan (spec 0.2). A target with no entry here is either met (see
`docs/perf-report.md`) or not yet measured (see `PROGRESS.md`).

Until `docs/perf-report.md` exists (WP-32), the numbers measured so far are recorded in "Measured"
below. Every number there was measured on one machine, called **PC-1**: Intel Core i7-10700K
(8 cores, 16 threads), 32 GB RAM, NVIDIA GeForce RTX 4060 (driver 621.01, Vulkan backend), Windows 11
Home 10.0.26200, Rust 1.99, release profile unless stated. Whether PC-1 stands for Tier M is not
decided (`docs/blockers.md` B-003).

## Deviations

### D-001: wgpu allocates in the steady-state frame loop `[PERF-R1]` `[AC-WP04-3]`

- **Measured.** `humanart-gpu` test `tests::ac_wp04_3_steady_state_frames_allocate_nothing_in_this_crate`
  (600 frames through the uniform ring, dirty uploads, cache lookups, profiler, readback ring,
  streaming and the loss check), PC-1, `cargo test -p humanart-gpu -- --nocapture`, dev profile:
  124 891 allocator events in total, **0 in `humanart-gpu`**, all 124 891 inside wgpu calls (about
  208 per frame).
- **Reason.** wgpu allocates inside its own per-frame calls (encoders, submissions, maps); that code
  is not ours and cannot be pre-sized from outside. The test books every allocation made inside a
  wgpu call separately (`into_wgpu`), so our own code is held to zero.
- **Plan.** Keep the crate at zero; minimise wgpu calls per frame (one queue write per ring, cached
  bind groups). Revisit if a later wgpu release offers allocation-free submission.

### D-002: JPEG decode is about 10 % short of 100 MP/s for 4:4:4 and detailed 4:2:0 files `[AC-WP03-3]`

- **Closed in round 6 (perf-gaps agent, not rerun by an integrator).** The IDCT is written so LLVM
  vectorises it, the float-to-byte conversion is branch-free, the bit reader loads 8 bytes at once
  when no 0xFF is among them and the Huffman lookup is 10 bits (`crates/io/src/jpeg/`; output bytes
  bit-identical to the previous transform, tests `eight_lane_form_equals_the_scalar_transform_exactly`
  and `branch_free_bytes_equal_the_saturating_cast`). `jpeg_throughput`, 4096x2048, one thread, PC-1
  40-100 % loaded by other agents, 3 runs: 4:2:0 q90 178-181 MP/s, 4:4:4 q90 127-135 MP/s, noisy
  4:2:0 q90 (3.24 bit/px) 129-139 MP/s: all above 100 MP/s. Whether PC-1 is Tier M stays open
  (B-003). The text below is the deviation as it stood.

- **Target.** JPEG decode >= 100 MP/s (`[AC-WP03-3]`, a target, Tier M CPU).
- **Measured.** `jpeg::tests::jpeg_throughput` (ignored test, `cargo test --release -p humanart-io
  jpeg_throughput -- --ignored --nocapture`; the test encodes its own images), 4096x2048, one
  thread, PC-1 about 20 % loaded by other agents: 4:2:0 quality 90 **139 MP/s** (met); 4:4:4
  quality 90 **88 MP/s**; a noisy 4:2:0 quality 90 file at 3.24 bit/px **90-93 MP/s** (100-116 MP/s
  to the component planes, before upsampling and colour conversion). Source:
  `target/agents/io-jpeg/STATUS.md`; not rerun by an integrator.
- **Reason.** Not analysed beyond the measurement. Entropy decoding grows with the bit rate (dense
  files) and 4:4:4 has twice the chroma samples of 4:2:0. The io-jpeg agent's last run made
  upsampling branch-free and batched the YCbCr conversion; the gain could not be isolated because
  the earlier measurement (4:2:0 63-70 MP/s) was taken at 100 % machine load.
- **Plan.** First remeasure on an idle PC-1 (the shortfall is within the load effect); if still
  short, profile the Huffman decoder. Whether PC-1 is Tier M is open (B-003).

### D-003: Rebuilding the rest mesh on the CPU after a landmark solve takes 4.5 ms `[LM-03]`

- **Met in round 6 on the measured surface (perf-gaps agent, not rerun by an integrator).** Half
  decoding no longer produces subnormal floats, contiguous components skip the per-entry index and
  `eval::evaluator::accumulate_all` adds runs of contiguous components block by block (bit-identical:
  `perf_accumulation_gives_the_same_bits_as_the_scaled_subnormal_form`). Same surface (10 201
  vertices, 108 dense components, 8 targets), release: the rebuild alone 4.86-4.92 ms before and
  1.56-1.75 ms after (`d003_rest_rebuild_before_and_after`, medians of 40, 3 runs); solve + apply +
  rebuild about 1.53-1.78 ms (`lm_03_latency_with_eight_constraints`, 8 runs, machine about 10 %
  loaded): under 2 ms with a 12-25 % margin. The gain from fusing needs components with contiguous
  vertex ranges; sparse regional components (as on the generated template) take the indexed path
  with only the subnormal fix. Not measured on the generated template: remeasure there before
  calling it met for the hero character. The text below is the deviation as it stood.

- **Target.** Landmark drag solve + apply < 2 ms CPU for <= 8 constraints (`[LM-03]`, 14.2).
- **Measured.** `eval::landmarks::tests::lm_03_latency_with_eight_constraints` (eval-cpu agent, not
  rerun by an integrator; PC-1), 10 201 vertices, 108 dense components, 8 targets, median of 20:
  solve 0.028 ms (dev profile) / 0.025 ms (release); solve + apply to the parameters 0.029 / 0.025 ms:
  **met**. The CPU re-accumulation of the rest mesh afterwards (every component changes: 1.1 M delta
  entries) is **5.2 ms dev / 4.5 ms release**.
- **Reason.** A landmark solve changes every component it spans, so the incremental one-slider path
  (`[EVAL-03]`) does not apply and the full delta accumulation runs. In the editor the mesh update is
  the GPU deformer's job (WP-10, ADR-031): full re-evaluation 0.30 ms GPU on a 40 803-vertex
  hero-scale synthetic surface (Measured below).
- **Plan.** Read `[LM-03]`'s "apply" as the parameter change (met) and leave the rest mesh to the GPU
  path; the CPU path stays the reference. Recorded as a deviation because the eval-cpu agent asked for
  it in case "apply" is read as including the CPU mesh update. Not yet measured end to end in the
  editor (the deformer is not wired into the renderer, `PROGRESS.md` rows 10 and 11).
- **State (2026-11-19).** The deformer is now wired into the renderer (`GpuScene::attach_deformer`,
  skin-shading agent) and the shell's viewer drives it, but the shell has no landmark drag yet
  (`[VP-02]` gizmos are not done), so the end-to-end number is still not measured.

### D-004: A slider drag through `Session::execute` allocates every frame `[PERF-R1]`

- **Target.** 0 steady-state heap allocations per frame (`[PERF-R1]`, 14.2).
- **Measured.** Not counted; reported by the commands agent from the code path: each command builds
  its edit list (resolve), runs the lock check and describes the outcome, all of which allocate.
- **Reason.** A command is a document transaction with an undo entry, a journal record and an event;
  it is not built for the frame loop.
- **Plan.** The application drags on the evaluation snapshot (zero-allocation evaluate, Measured
  below) and sends commands at its own rate; consecutive commands of the gesture coalesce into one
  undo step (`[CMD-03]`) and their journal frames are batched (100 ms by default). Not yet verified:
  the application shell (WP-21) does not exist.
- **State (2026-11-19, WP-21 part 1).** The shell exists. A steady `App::frame` (1600x900, all
  panels, no input) makes 0 allocations (app-shell agent's test), and the engine's `Model::sync`
  makes 0 when the document did not change; slider gestures become one undo step. The frames of a
  drag were not counted, so whether the drag itself stays off the allocator is still not shown.
- **State (round 6, perf-gaps agent; not rerun by an integrator).** Narrowed to the journal batch
  write. `Session::drag_scalar` folds a drag frame into the open gesture in place (same document,
  history, events, revision and journal bytes as `execute`:
  `cmd03_perf_r1_drag_frames_equal_execute_and_fold_in_place`, 2221 frames). The app test
  `perf_r1_slider_drag_frames_do_not_allocate` drags a Properties slider through `App::frame` (UI,
  intents, session, engine sync with CPU evaluate): 200 measured frames, **0 allocations, 0
  reallocations**, 200 model re-evaluations (before: about 25.6 allocator calls per frame). With a
  project open and the default 100 ms journal batch, 74 of 300 frames of 16 ms allocate (1484
  calls, worst 38; `perf_r1_steady_drag_frames_do_not_allocate`): the batch flush (JSON record
  encode + file append, about 20 calls) and the first frame of the next batch. That is I/O every
  100 ms, not per frame; it stays as this deviation until the owner says whether `[PERF-R1]` covers
  it.

### D-005: The GPU bakers need about 68 bytes of GPU memory per target texel

- **Target.** No 14.2 target covers baking (an offline tool, `humanart-tools`). Recorded because it
  limits the tile size a given GPU can bake: `[MOD-04]` asks for >= 10 texels/mm on the face tile, and
  the bakers agent names an 8192² face tile.
- **Measured.** The bakers agent's figure, not rerun: 68 B per texel of the target. Its STATUS gives
  "1.1 GB at 4096², 4.4 GB at 8192²"; 68 B x 8192² is 4.56 GB (4.25 GiB) and 68 B x 4096² is 1.14 GB
  (1.06 GiB), so the 8192² figure in the STATUS does not follow exactly from the per-texel number.
  Whether an 8192² bake fits on PC-1's 8 GB RTX 4060 was not tested (the template bakes ran at 1024²).
- **Reason.** The UV G-buffer and the per-texel outputs of every map are resident at once for the
  whole tile.
- **Plan.** Not decided. Options: bake large tiles in sub-tiles (the ray kernels are already
  dispatched in row bands of <= 2^24 rays), or require a large GPU for 8192² bakes.

### D-006: Surfaced skin geometry + shading reaches the 2.0 ms budget only at 38 % face coverage `[PERF-R6]`

- **Target.** Skin geometry + shading passes (full face visible) <= 2.0 ms (14.2).
- **Measured.** `crates/render/tests/surface.rs::perf_surfaced_skin_geometry_at_1440p` (skin-render
  agent, not rerun by an integrator), RTX 4060 shared with other agents, 2560x1440, the template
  face covering 38 % of the pixels, plain and surfaced rendered alternately, fastest frame of 120:
  geometry pass plain 1.627 ms, surfaced (ADR-052) **1.992 ms** (surface cost 0.365 ms, median pair
  difference also 0.365 ms; micro-detail about 0.17, compositor fetches and the rest about 0.2,
  coat + sheen about 0.03 ms); subsurface 0.58-0.59 ms. Met at that coverage with 0.008 ms to
  spare. Not measured with the face filling the whole frame: the agent reads the plain path alone
  as about at the limit there, so the surfaced pass would exceed 2.0 ms. The test asserts the
  surface's own cost <= 0.5 ms and SSS <= 1.2 ms; the 2.0 ms total is printed, not asserted.
- **Reason.** Shading cost grows with the skin pixels; the surfaced path adds compositor fetches,
  micro-detail sampling and the coat and sheen lobes to every skin pixel.
- **Plan (the agent's, not done).** One shade evaluation for both specular lobes for directional
  and spot lights, a depth prepass so occluded fragments are not shaded, no micro-detail sampling
  past mip ~5 (use the exemplar's total variance). Whether "full face visible" means the face in
  view or the face filling the frame is for the owner to say (the same question as the SSS note
  below).

## Measured

| What | Number | Command / test | Measured by |
|---|---|---|---|
| PNG decode, 4096x2048 smooth + noise, RGBA8 / RGB8 / RGBA16 (target `[AC-WP03-3]` >= 150 MB/s output on Tier M) | 275-280 / 251-252 / 262 MB/s output | `cargo test --release -p humanart-io --features png png_throughput -- --ignored --nocapture` (best of 3, nothing else running) | integrator, 2026-11-19 |
| PNG encode, same images, Fast / Default | RGBA8 35 / 15, RGB8 35 / 15, RGBA16 32 / 33 MB/s; size 70.1 / 68.8 % (8-bit), 90.0 % (16-bit) of raw | same | integrator, 2026-11-19 |
| DEFLATE, 32 MiB lightly mutated fixture text, Store / Fast / Default / Best | encode 1259 / 94 / 19 / 8 MB/s, decode 1911 / 637 / 708 / 727 MB/s output; size 100 / 44.7 / 42.0 / 41.8 % | `cargo test --release -p humanart-io --features deflate deflate_throughput -- --ignored --nocapture` (ran alongside the LZ measurement) | integrator, 2026-11-19 (the io-png agent measured within 5 %) |
| LZ codec, 32 MiB inputs, 256 KiB blocks, 1 thread; decompress includes the checksum | text: ratio 0.571, 318 / 1133 MB/s; sparse: 0.327, 631 / 1346 MB/s; mesh floats and random: stored, ~2.9 / ~4.3 GB/s (compress / decompress) | `cargo test --release -p humanart-io --features lz measure_speed -- --ignored --nocapture` (ran alongside the DEFLATE measurement) | integrator, 2026-11-19 (io-json agent: 327 / 1234 MB/s on text) |
| OBJ streaming read, 3.9 MB generated grid mesh, 64 KiB chunks | 251 MB/s | `obj::tests::large_file_throughput`, release, other tests running in parallel | io-ttf agent (not rerun) |
| Octahedral normal / tangent encoding, worst angular error over 200 000 random directions | oct16 0.00245 deg, oct8 0.627 deg, tangent 0.0042 deg | `linalg::pack` tests | math-linalg agent (not rerun) |
| PCG on a 32x32 badly scaled 2D diffusion problem, tolerance 1e-10 | plain CG 5935 iterations, Jacobi 252, IC(0) 79 | `solve::pcg::tests::ac_wp02_1_preconditioners_reduce_iterations_on_a_badly_scaled_problem` | math-solve agent (not rerun) |
| GPU timestamp profiler | 4096-invocation, 20 000-iteration LCG compute pass about 0.50 ms; empty pass 64 ns | `humanart-gpu` profiler tests | gpu agent (not rerun) |
| GPU readback ring, 12 frames of 4 ms, 4 slots | 10 readbacks delivered by non-blocking polls during the frames, 1 refused while all slots were busy | `humanart-gpu` readback tests | gpu agent (not rerun) |
| JPEG decode, 4096x2048, one thread, PC-1 ~20 % loaded (target `[AC-WP03-3]` >= 100 MP/s) | 4:2:0 q90 139 MP/s; 4:4:4 q90 88 MP/s; noisy 4:2:0 q90 (3.24 bit/px) 90-93 MP/s (100-116 to planes) | `jpeg::tests::jpeg_throughput`, release, `--ignored` | io-jpeg agent (not rerun); superseded by the round-6 row below |
| JPEG decode after the round-6 IDCT and bit-reader changes, same images, PC-1 40-100 % loaded, 3 runs | 4:2:0 q90 178-181 MP/s; 4:4:4 q90 127-135 MP/s; noisy 4:2:0 q90 129-139 MP/s (183-192 to planes); upsampling + colour 15-20 ms of 60-70 ms per image; IDCT alone 98-114 ns per block (old form 223-234 ns) | `cargo test --release -p humanart-io --features jpeg jpeg_throughput -- --ignored --nocapture` | perf-gaps agent (not rerun); D-002 closed |
| JPEG encode, same images | 4:2:0 76-79 MP/s, 4:4:4 35-55 MP/s | same | io-jpeg agent (not rerun) |
| JPEG encoder PSNR, 256x192 test texture, 4:4:4 / 4:2:0 / grey | q50 36.1 / 34.8 / 37.3 dB, q75 37.3 / 35.7 / 38.8, q90 38.9 / 37.0 / 41.1, q100 50.6 / 38.6 / 58.9; 4:2:0 files ~30 % smaller than 4:4:4 | `jpeg/tests.rs` encoder round trips | io-jpeg agent (not rerun) |
| LOD generation `[MOD-05]`, cube-sphere 99 846 vertices (98 306 welded), 196 608 triangles, global + local deltas, 2-joint skin | four levels in 6.35 s; LOD1 98 304 tris, geometric max 1.9e-5 m, delta max 1.2e-5 m; LOD2 49 152, 3.7e-5 / 1.5e-5; LOD3 23 592, 9.6e-5 / 5.2e-5; LOD4 9 830, 2.3e-4 / 1.3e-4; skin L1 max <= 0.002 | `cargo test --release -p humanart-model --lib mod_05_benchmark -- --ignored --nocapture` | lod-gen agent (not rerun) |
| LOD generation, sphere 24 (3 458 welded vertices), LOD1..4 at 50 / 25 / 12 / 5 % | geometric max 0.37 / 1.0 / 1.7 / 6.2 mm, delta max 0.18 / 0.47 / 0.86 / 0.79 mm; default budgets 0.5 / 1.5 / 4 / 10 mm all met | `simplify` tests | lod-gen agent (not rerun) |
| BC7 / BC6H GPU encode, 4096x4096, warm pipelines, mean of 3 (`[AC-WP28-2]`; the spec names no threshold) | BC7 fast 18.7 ms, normal 163 ms, high 788 ms; BC6H UF16 fast 18.2 ms, normal 204 ms, high 366 ms | `humanart-render` `compress` bench (ignored test), release | texture agent (not rerun) |
| BC7 PSNR, fixed-seed synthetic 256x256 content, fast / normal / high | skin 54.59 / 56.60 / 58.35 dB; gradients + noise 48.72 / 50.74 / 51.38; alpha shapes 51.98 / 52.36 / 53.34; normal + roughness 41.41 / 42.64 / 46.55; uniform random RGB 13.98 / 16.70 / 18.26 | `compress` tests | texture agent (not rerun) |
| BC7 normal maps, mean / max angular error, xyz \| xy + rebuilt z | fast 1.48 / 11.9 \| 1.48 / 12.1 deg; normal 1.27 / 6.3 \| 1.24 / 6.6; high 0.83 / 3.8 \| 0.81 / 3.8 | `compress` tests | texture agent (not rerun) |
| BC6H PSNR in the log2(1 + x) domain, fast / normal / high | HDR sky UF16 68.77 / 69.17 / 69.24 dB; signed fields SF16 39.38 / 41.63 / 41.74; skin as linear float UF16 53.59 / 54.35 / 54.45 | `compress` tests | texture agent (not rerun) |
| GPU mip chains vs the CPU reference `[IO-05]` | <= 1 step for 8-bit sRGB, alpha, data, linear, normal + roughness; float < 2e-3 relative | `compress` tests | texture agent (not rerun) |
| UI frame: build + layout + record of 600 widgets with text and semantics (part of `[UI-10]`; the 2000-row list, docked viewports and UI GPU time need docking) | median 0.125 ms per frame, 0 allocations once warm, 1 draw batch | `ui::tests::perf_r1_steady_frames_do_not_allocate`, test (debug) profile | ui-core agent (not rerun) |
| Font bake, 5 faces (Inter x3, JetBrains Mono, Lucide) | 10.5 s release; `.anfont` regular 372 761 B, semibold 410 083 B, italic 401 911 B, mono 248 546 B, icons 1 340 274 B; atlas 6 pages of 1024^2 RGBA8 = 24 MiB | `tools/fontbake/bake.sh` | ui-core agent (not rerun) |
| WGSL library vs CPU (RTX 4060, Vulkan) | tone mappers worst \|gpu - cpu\| 1.2e-7 (PBR Neutral), 4.2e-7 (ACES fitted); naga 6 shaders / 19 variants, device 41 pipelines at first, later 7 shaders / 21 variants / 43 pipelines with `ui/quad.wgsl` | `humanart-shaders` `tests/library.rs`, `tests/validate.rs` | shaders agent (tests rerun by integrator2 2026-11-19: 24 + 12 + 3 pass; counts not printed without `--nocapture`) |
| CPU evaluation `[EVAL-01]`, 40 401 vertices, 72 components, 3 joints | full evaluate 7.6 ms; one-slider incremental rest update 0.22 ms; 0 allocations per evaluate | `humanart-model` `eval` tests, dev profile (optimised) | eval-cpu agent (not rerun) |
| Landmark solve `[LM-03]`, 10 201 vertices, 108 components, 8 targets, median of 20 | solve 0.028 / 0.025 ms (dev / release), solve + apply 0.029 / 0.025 ms; CPU rest rebuild 5.2 / 4.5 ms (D-003) | `eval::landmarks::tests::lm_03_latency_with_eight_constraints` | eval-cpu agent (not rerun) |
| GPU deformation `[AC-WP10-2]`, hero-scale synthetic surface (40 803 v, 80 000 tris, 323 components of which 131 shape, 150 expressions, 24 joints, 4 influences), release | full re-evaluation 0.30 ms GPU (target 0.6: met; accumulate 0.18, skin 0.05, normals 0.05, strain 0.03); one slider 0.093 ms GPU (target 0.15: met; scatter 0.004, skin 0.025, normals 0.044, strain 0.025); CPU update + encode 13.6 µs (target 100 µs: met; 84 µs in the dev profile) | `deform::tests::ac_wp10_2_deformation_timings`, `--release -- --nocapture` | gpu-eval agent (not rerun) |
| GPU deformation parity and drift `[AC-WP10-1]` `[AC-WP10-3]` | positions 4.6e-7 m (tolerance 2.5e-5), normals/tangents 0.028° (0.5°), 0 handedness mismatches, edge strain 8.6e-6, region strain 3.7e-7 (strain tolerance 1e-3 chosen by the agent, the spec gives none); drift after 10 000 slider moves 2.5e-7 m with periodic refresh, 3.2e-6 m without | `deform::tests::ac_wp10_1_*`, `ac_wp10_3_*` | gpu-eval agent (not rerun) |
| GPU deformation allocations `[PERF-R1]` | `plan` (CPU half of update) 0; update + encode incl. wgpu internals 236 allocator events per 8 frames (wgpu staging and command recording, D-001) | `deform::tests::perf_r1_update_plan_allocates_nothing` | gpu-eval agent (not rerun) |
| Core render passes `[PERF-R6]`, material test scene (24 629 v, 46 560 tris), Review, TAA, median of ~120 frames | 1920x1080: GPU 1.34 ms (geometry 0.95, TAA 0.225, composite 0.096, shadows 0.046, tonemap 0.030), CPU encode + submit 0.15-0.18 ms; 2560x1440: GPU 2.07 ms (geometry 1.40, TAA 0.37, composite 0.21, shadows 0.044, tonemap 0.052), CPU 0.16-0.25 ms. Not the hero character: no skin, SSS, hair or eye passes yet | `humanart-render` `tests/timings.rs::perf_r6_core_pass_timings_at_1080p_and_1440p` | render agent (not rerun) |
| 8K still `[AC-WP11-3]`, 7680x4320, 4 spp, material test scene | untiled 0.17-0.21 s; 12 tiles (2048 + 32 overlap) 0.26-0.31 s | `tests/stills.rs::ac_wp11_3_tiled_8k_still_equals_the_untiled_render` | render agent (not rerun) |
| UI benchmark scene `[UI-10]` `[AC-WP12-1]`, 1920x1080, scale 1, 30 warm-up + 120 frames | release: UI build + layout + draw list median 0.25 ms (target 0.5: met), upload + record 0.025 ms; GPU time of the UI pass median 0.196 ms (target 0.3: met; p95 0.199 in the test profile); 1 draw call, 801 instances, 0 allocations. Test profile: build 0.78 ms (p95 1.07), upload + record 0.17 ms | `bench::tests::ac_wp12_1_ui_benchmark_scene_records_its_contract`, `cargo test --release -p humanart-ui` | ui-widgets agent (the test passes in the integrator3 rerun; the numbers were not reprinted) |
| Skin partial-rect re-resolve `[AC-WP13-3]`, 2048 tile, 6 layers, GPU timestamps, median of 7 | 32² 0.032 ms; 64² 0.033; 128² 0.052; 256² 0.161; 512² 0.594; 1024² 2.330; 2048² 8.874 ms (2.46 / 2.27 / 2.22 / 2.12 ns per texel from 256² up) | `skin::tests::ac_wp13_3_partial_rect_cost_scales_with_area` | skin agent (not rerun) |
| Skin compositor GPU vs CPU | max \|diff\| planes 4.9e-4, albedo 4.9e-4, scatter 9.8e-4 mm, flush 6.1e-5, specular 2.2e-3; `prepare` 0 allocator events over 50 frames | `humanart-render` `skin` tests | skin agent (not rerun) |
| Bakers, generated template (63 029 v, 112 864 tris), RTX 4060 | 1024² thickness tile, 64 rays, 16 padding passes + readback about 180 ms; CPU vertex curvature 46 ms; GPU target memory 68 B/texel (D-005) | `bake::tests::ac_wp15_2_thickness_on_the_template` | bakers agent (the test passes in the integrator3 rerun; timings not reprinted) |
| SSS separable passes `[AC-WP14-3]`, 2560x1440, Review, template face filling the frame height at 16:9 (58 % skin pixels), median of 60 frames | `sss_h` + `sss_v` 0.79-0.83 ms (target 1.2 ms: met); whole frame 3.1 ms. The agent's linear estimate for a frame 100 % skin: about 1.4 ms (not measured). Before the one-fetch-per-tap change: 1.34 ms at 33 % skin | `tests/skin.rs::ac_wp14_3_subsurface_passes_at_1440p` | skin-shading agent (the lead's rerun of `tests/skin.rs` passes; numbers not reprinted) |
| Skin shading energy and kernel `[AC-WP14-1]` | skin furnace min / max / mean: Review 0.9985 / 1.0000 / 0.9992, Interactive 0.9995 / 1.0000 / 0.9996 (bound: max < 1.015); separable passes vs the CPU reference on a hard shadow corner: worst 0.28 of peak 219 (0.13 %, bound 1 %); backlit 6 mm disc red 32.1 vs 32.8 from the profile's `T(5.5 mm)`; past the terminator (4°) red 0.0037 / 0.0021 of the lit side (Review / Interactive), plain material exactly 0 | `tests/skin.rs` | skin-shading agent (lead rerun: passes) |
| Deformed-mesh motion vectors `[EVAL-05]` | worst error 0.0017 px for motions up to 5.6 px (bound 0.02 px); zero motion once still | `humanart-render` `tests::eval_05_deformed_mesh_is_pulled_from_the_deformer_with_correct_motion` | skin-shading agent (inside the lead's render lib rerun, 87 pass) |
| Groom GPU vs CPU `[AC-WP17-2]`, 20 000 strands x 16 points | shape max 3.4e-6 m; stored (f16) max 6.1e-5 m, 0 components beyond 2 f16 ulp; attach max 6.1e-5 m | `groom/tests.rs::ac_wp17_2_gpu_modifiers_match_the_cpu_reference` | hair agent (lead rerun: render groom 5 pass) |
| Groom edit latency `[AC-WP17-2]`, 100 000 strands x 16 points, 1000 guides, median of 20 comb edits of 50 guides | regenerate 1.14 ms GPU, attach 0.15 ms GPU, guide edit CPU + upload 0.49 ms; a density / count edit: CPU placement 337 ms (optimised test profile) + layout upload 11.8 ms. The test asserts regenerate + CPU < 16.7 ms (one 60 fps frame, the agent's bound: 14.2 has no groom budget) | `groom/tests.rs::ac_wp17_2_groom_edit_latency_100k_strands` | hair agent (lead rerun: passes; numbers not reprinted) |
| Hair cards vs strands `[HAIR-10]`, 30 000 strands `long_straight`, side view 2.5 mm/px, 256 / 96 / 32 cards | soft IoU 0.827 / 0.804 / 0.783; coverage ratio 0.84 / 0.82 / 0.81 | `groom/tests.rs::hair_10_card_lods_bake_export_and_match_the_strands` (asserts LOD0 IoU > 0.5) | hair agent (lead rerun: tools groom 5 pass) |
| Expression build on the generated template | 1.97 s dev / 1.60 s release; 118 delta components, about 188 000 entries; 0 entries outside their region, 0 seam mismatches | `expression::build` tests | expression agent (lead rerun: 26 expression tests pass) |
| Expression sweep `[AC-WP19-1]`, 89 controls x 4 steps, and 34 presets | 0 inverted triangles; max displacement 32.8 mm (AU27 chin); area ratio within [0.2, 5] (authored bounds; triangles < 0.1 mm² and the eye overlay parts skipped) | `expression::sweep` tests | expression agent (lead rerun: passes) |
| Wrinkle drivers `[EXP-04]` `[AC-WP19-2]` | fold >= 0.5 at forehead (AU01 0.84, AU02 0.56), glabella (AU04 0.89), crow's feet (AU06 0.86), nasolabial (AU12 0.73, AU11 0.61); neutral < 0.01 | `expression::controls` tests | expression agent (lead rerun: passes) |
| Application shell frame `[PERF-R1]`, 1600x900, all panels, steady | 0 allocations | `crates/app` tests | app-shell agent (lead rerun: 17 app tests pass, 18 after the fixes round) |
| GPU device lifecycle stress (ADR-040), 10 rounds x 7 parallel test binaries, cap 600 s | 0 hangs, 0 timeouts; longest run 66 s (render lib under load; 14 s alone) | `target/agents/fixes/stress/stress.sh 10 600` | fixes agent (not rerun) |
| Randomised PCA `[STAT-02]` vs exact PCA, N 1500, D 120, rank 6 + noise | variance gap 6.6e-16, component dot products 1 within 1e-9, projections within 1e-8; slow 1/i spectrum, k 10: variance gap 1.1e-6 | `solve::pca` tests | fixes agent (lead rerun: math 255 pass) |
| LOD generation `[MOD-05]` `[AC-WP28-1]` on the generated template (114 400 LOD0 triangles, 57 272 welded vertices, no deltas), after ADR-045 | LOD1-4 triangle ratio 0.500 / 0.250 / 0.120 / 0.050; geometric max 0.499 / 1.393 / 3.341 / 9.378 mm (budgets 0.5 / 1.5 / 4 / 10); feature max 0.495 / 1.468 / 3.588 / 8.484 mm; skin L1 max 0.098 / 0.198 / 0.339 / 0.496 (budgets 0.1 / 0.2 / 0.35 / 0.5), mean 0.002 / 0.007 / 0.016 / 0.047; landmarks max 0.41 / 1.20 / 2.15 / 6.64 mm; refused collapses 58 / 49 / 50 / 44, undone 0. Four levels 7.6 s (one run; the shared PC varies 1.5x) | `mod_05_generated_template_meets_level_targets` (ignored), release | lod-fix agent (lead rerun in round 5: all four levels pass; numbers not reprinted) |
| LOD generation, 100k cube-sphere benchmark, after ADR-045 | 12.0-19 s under load (6.35 s in the lod-gen agent's row above, before the change); identical outputs; about 2x from verification and the UV grid (7.4 s with both disabled, 8.9 s with the UV grid only) | `mod_05_benchmark` (ignored), release | lod-fix agent (not rerun) |
| Wrinkle blending `[SKN-30]` GPU vs CPU | driver activations within 1e-6; blended texels max \|diff\| < 1e-5 against the CPU twin on f16-rounded texels; `DriverTable` 0 allocations per frame | `humanart-render` `skin::wrinkle` tests, `skin_appearance::wrinkle` tests | skin-finish agent (not rerun) |
| GPU-to-pool page copy `[SKN-03]`, 20 pages per channel | mip 0 bit-exact; mip 1 max \|diff\| 2.4e-4 (one f16 step); specular RGBA8 exact | `skn03_gpu_to_pool_copy_streams_the_resolved_set` | skin-finish agent (not rerun) |
| Skin colour vs ISSA `[AC-WP13-1]` (ADR-047), CIEDE2000 after fitting melanin, eumelanin ratio and haemoglobin, specular term added; before = old spectra and mapping under the same comparison | ITA bins very light / light / intermediate / tan / brown / dark: before 0.70 / 0.21 / 0.57 / 1.48 / 3.45 / 5.56, after 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00; L* bands 25-75 worst before 6.89 (25-30), after 0.13 (25-30); 12 body sites max 2.72 before, 0.60 after (ring finger); 36 decile-edge swatches max 7.83 before, 2.14 after (very light, a* high); spectral RMS per ITA bin before 0.036 / 0.032 / 0.031 / 0.031 / 0.037 / 0.022, after 0.027 / 0.031 / 0.026 / 0.019 / 0.011 / 0.008. Tolerances (<= 1.0, edges <= 2.5, RMS < 0.035) chosen by the agent, the edge and RMS ones after seeing the results | `pigment::tests::ac_wp13_1_reproduces_the_issa_reference_swatches`, `ac_wp13_1_fitted_spectra_follow_the_measured_spectra` | skin-calib agent (lead rerun: 62 `skin_appearance` tests pass, release; numbers not reprinted) |
| Identity components `[ID-07]`, generated template, largest displacement per unit weight | height 37.7 mm (head top; soles fixed), leg length 31, arm length 22, mass 15.7, shoulder width 13.9, hip width 11.1, sex 8.0 mm; face components 0.5-3 mm, teeth 0.35-1.4 mm | identity tests (`target/agents/identity/previews/components.txt`) | identity agent, round one (not rerun) |
| Identity region borders `[ID-02]`, largest dihedral change | each component alone at weight 2: border <= 1.5 x interior + 5 deg for all; random identities (6 seeds, components uniform in [-2, 2]): 99.9 % quantile border 34-57 deg vs interior 27-45 deg; maxima 120-166 deg on sliver triangles (0.2-1 mm²) at the lip corners, alar crease and A-pose armpit folds | `id_02_no_normal_discontinuity_at_region_borders` | identity agent, round one (not rerun) |
| Landmark solve + CPU rest rebuild `[LM-03]` after the round-6 evaluator changes, same surface | rebuild 1.56-1.75 ms (before 4.86-4.92 ms, same run); solve + apply 0.024-0.025 ms; solve + apply + rebuild about 1.53-1.78 ms | `d003_rest_rebuild_before_and_after` (ignored), `lm_03_latency_with_eight_constraints`, release | perf-gaps agent (not rerun); D-003 |
| Slider drag through the app `[PERF-R1]` `[CMD-03]`, 400 warm-up + 200 measured frames | 0 allocations, 0 reallocations, 200 model re-evaluations (before: 3077 allocator calls in 120 frames); with a project and 100 ms journal batches: 74 of 300 frames allocate (1484 calls, worst 38) | `crates/app` `perf_r1_slider_drag_frames_do_not_allocate`; doc `perf_r1_steady_drag_frames_do_not_allocate` | perf-gaps agent (not rerun); D-004 |
| BC7 / BC6H / mip GPU time from timestamp queries `[PERF-R6]` `[AC-WP28-2]`, 4096x4096, warm pipelines, median of 5 (GPU / wall) | BC7 fast 6.36 / 6.7 ms, normal 170.1 / 170.4, high 789.7 / 789.9; BC6H UF16 fast 19.6 / 19.9, normal 212.3 / 212.6, high 455.2 / 455.6; mip chain 12 levels RGBA8 sRGB 0.76 / 2.1, RGBA16F 1.28 / 2.6 ms | `cargo test -p humanart-render --release --lib -- --ignored --nocapture gpu_timestamps` | perf-gaps agent (not rerun) |
| Surfaced skin geometry pass (ADR-052), 2560x1440, template face 38 % of the pixels, fastest of 120 frames | plain 1.627 ms, surfaced 1.992 ms (surface 0.365 ms); subsurface 0.58-0.59 ms; whole 1440p frame 6.9 ms in `tests/timings.rs` (it failed once under GPU load at its 8 ms bound and passed on rerun) | `tests/surface.rs::perf_surfaced_skin_geometry_at_1440p` | skin-render agent (not rerun); D-006 |
| Micro-detail anisotropy `[SKN-20]` `[SKN-21]` | relative variance across / along the skin lines 0.0124 / 0.0061 (2.04x); with UVs rotated 90 deg 0.01246 / 0.00603; strength 0 gives 0 | `tests/surface.rs::skn20_micro_detail_follows_the_skin_lines_not_the_uvs` | skin-render agent (not rerun) |
| Lip zone mask `[SKN-11]`, 2048² head tile, mm² of surface weighted by the mask | before: total 3470, more than 2 mm outside the vermilion 2112, off the lip regions 69.5; after: total 1087 (incl. inner lips), outside 0.0, off the lip regions 5.2, vermilion core 1.00 | `zone_bake` `skn11_lip_mask_stays_on_the_vermilion` | skin-wire agent (not rerun) |
| Generated template build, release | whole template incl. LODs 12.97 s (surface ~0.1 s, UV ~1.4 s, landmarks 0.17 s, skin 0.075 s, LODs the rest); LOD0 63 029 vertices (37 938 body), 13 parts; package 10 631 125 bytes, write + read 75 ms. Texel density under the `[MOD-04]` target on 10.1 % of the face tile (median 20.2 texels/mm), torso 27.5 %, legs 42 %, hands-feet 22 %, mouth 26 % (warnings); packing coverage 0.21-0.60 | `generate` tests | template-gen agent, round 2 (not rerun) |
| Template against ANSUR II sex midpoints (`anat_02`) | stature, tragion to top of head, suprasternale, cervicale, trochanterion, lateral femoral epicondyle and lateral malleolus heights within 0.3 mm; off: biacromial -9.3 mm, acromial height +24.9, hip breadth +22.8, crotch height +48.0, foot length +4.8, hand length (relaxed) -5.9 mm | `anat_02_surface_meets_the_ansur_constants` (`target/agents/template-gen/previews/anthropometry.txt`) | template-gen agent (file as read at 20:06, already rewritten by its round 3) |
| Identity calibration `[ID-03]` (ADR-051), change per unit weight | 14 axes within 0.1 % of one SD (e.g. stature 66.4 mm, crotch height 45.6, biacromial 18.8, hip breadth 25.5, neck circumference 22.7); mass girths 97 / 60 / 101 / 52 mm against 106 / 84 / 76 / 58 expected, +13.9 L per unit | `id_03_calibrated_axes_move_their_measurement_by_one_sd` | identity agent, round two STATUS (not rerun; `previews/calibration.txt` has since been rewritten by a later run with other values) |
| Identity shape roughness `[ID-02]` | second-derivative measure 0.10-0.57 for all body shapes (limit 0.65); a glute shape quantised to 8 steps 1.21 | `id_02_body_shapes_have_no_stepped_falloff` | identity agent, round two (not rerun) |

No spec target exists for the DEFLATE, LZ, OBJ, LOD timing and encode numbers, nor for the
skin-reference ΔE (ADR-047 records it as an open owner question); they are recorded for
comparison. `[AC-WP28-2]` asks for BC7/BC6H numbers "against thresholds" but the spec sets none.

The 14.2 budget "skin resolve for a single painted-rect update <= 0.2 ms GPU" names no rectangle size:
the measured re-resolve is under it up to 256² (0.161 ms) and over it from 512² (0.594 ms). Whether a
512² rectangle is a single painted-rect update is for the owner to say; no deviation is filed until
then. That test's per-texel cost-ratio assertion fails on a shared GPU: in most loaded runs of the
skin-shading agent (ratio 2.79x) and in 13 of 20 runs of the fixes agent's stress loop; it passes
in the lead's rerun of the render lib (87 pass).

The 14.2 budget "SSS (separable, 1440p, face filling the frame) <= 1.2 ms" is met with the face
filling the frame height (58 % skin pixels); the skin-shading agent estimates about 1.4 ms if every
pixel were skin. Whether "face filling the frame" means the height or the whole frame is for the
owner to say; no deviation is filed until then.

`[AC-WP17-2]` asks for "groom edit latency within budget for 100k strands", but 14.2 has no groom
budget; the hair agent's test uses one 60 fps frame for a guide edit. A density, mask or count
edit reruns the CPU placement (337 ms) and is not held to a bound.

The skin-calib agent saw `perf_surfaced_skin_geometry_at_1440p` (`crates/render/tests/surface.rs`)
fail once while other agents were building and using the GPU; it did not investigate (timing does
not depend on albedo values). That test asserts the surface's own cost (<= 0.5 ms) and SSS
(<= 1.2 ms), not the 2.0 ms total; the total is deviation D-006.

Texture compression has no 14.2 budget and `[AC-WP28-2]` names no threshold: the GPU timestamp
numbers above are recorded, not judged. BC6H fast and every normal / high encode at 4096² take
longer than one 60 fps frame, so they stay editor-time or offline work (`[SKN-04]`: painting uses
the uncompressed mode) or must be split across frames if ever run interactively.
