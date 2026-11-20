# Reference conventions: atroxabot-app

Requirement: REF-01 (from the original specification, removed 2026-11-19). This document records what the team's existing
wgpu application does, so that HumanArt can follow it (REF-02) without anyone opening it again.

- Reference application: **AtroxaBot**, version 0.1.10 (`Cargo.toml`, `[workspace.package]`), commit
  `a1c5dc5` "Version 0.1.10".
- Location read: `C:\Users\jerry\Documents\atroxabot-app`. All paths below are relative to that root.
- Read in full: `crates/desktop/src/{main,app,gpu,paint,ui,text,font,atlas,raster,theme,color,platform,
  titlebar,log,image,form,editor}.rs`, `shader.wgsl`, `shell/widgets.rs`, `build.rs`, every `Cargo.toml`,
  `CLAUDE.md`, `rustfmt.toml`, `clippy.toml`, `crates/mcp/src/{error,http}.rs`, the `packaging/` scripts
  and manifests, `.github/workflows/ci.yml`. Read in part (the structure, drawing, input and
  protocol code; not the application's domain logic or every test): `crates/core/src/error.rs` (all but
  the end of its tests), `shell/mod.rs`, `shell/menu.rs`, `shell/snapshot.rs`, `shell/composer.rs`,
  `shell/thread.rs`, `doc.rs`, `orb.rs`, `logo.rs`, `tray.rs`, `README.md`,
  `crates/mcp/src/{lib,proto,conn,stdio}.rs`. Statements below about a partly read file are limited to
  the lines cited.
- wgpu API details were checked against `~/.cargo/registry/src/*/wgpu-30.0.1` and `wgpu-types-30.0.1`.
- "No equivalent" means the reference has nothing of the kind; per REF-02 the specification then governs.

The reference is a **2D UI application**: one window, one pipeline, one draw call, no depth buffer, no
3D, no compute, no web build. Its plumbing (window, device, frame pacing, input, text, UI toolkit,
logging, platform shims) is directly relevant; its renderer structure is much smaller than what
HumanArt needs. Section 19 lists the gaps.

---

## 1. Instance, adapter, device, surface

All in `crates/desktop/src/gpu.rs`, `Renderer::new` (lines 141-201). It is an `async fn` driven by the
hand-written `block_on` (`app.rs:2037`).

| Item | What the reference does | Evidence |
|---|---|---|
| Instance | `wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)))`, where `display` is winit's `OwnedDisplayHandle` | `gpu.rs:142-143`, `app.rs:505` |
| Backends | Not chosen in code. The descriptor's default is `Backends::all()`, overridable by the `WGPU_BACKEND` environment variable (that is what `_from_env` does). Compiled-in backends are set by cargo features: `dx12`, `metal`, `vulkan`, `gles` | `Cargo.toml:21` |
| Surface | `instance.create_surface(window)` with `window: Arc<Window>`, giving `wgpu::Surface<'static>` | `gpu.rs:145`, `gpu.rs:104` |
| Adapter | `power_preference: wgpu::PowerPreference::LowPower`, `compatible_surface: Some(&surface)`, rest default. Reason given: "a UI does not need the discrete card and waking it costs battery and startup time" | `gpu.rs:136-153` |
| Features | None requested (`..Default::default()`, so `required_features` is empty) | `gpu.rs:155-160` |
| Limits | `wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())`: the WebGL2 floor (no storage buffers, no compute, 2048 px textures) with only the three `max_texture_dimension_*` limits raised to the adapter's | `gpu.rs:157` |
| Memory hints | `wgpu::MemoryHints::MemoryUsage` | `gpu.rs:158` |
| Device label | `Some("atroxabot")` | `gpu.rs:156` |
| Surface format | The first format in `caps.formats` that is **not** sRGB (`!f.is_srgb()`), else the first format, else `GpuError::NoSurfaceFormat`. Reason: blend in sRGB space "like browsers do" | `gpu.rs:177-186` |
| Present mode | `wgpu::PresentMode::AutoVsync` | `gpu.rs:193` |
| Alpha mode | `caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto)` | `gpu.rs:195` |
| Other config | `usage: RENDER_ATTACHMENT`, `color_space: wgpu::SurfaceColorSpace::Auto`, `desired_maximum_frame_latency: 2`, `view_formats: Vec::new()`, width and height `.max(1)` | `gpu.rs:187-197` |
| Error handler | `device.on_uncaptured_error(Arc::new(..))` logs the first 8 errors (`MAX_LOGGED_ERRORS`) instead of panicking | `gpu.rs:98-100`, `163-169` |
| Device lost | `device.set_device_lost_callback(..)` logs and sets an `Arc<AtomicBool>` | `gpu.rs:170-175` |

The cargo feature set is `default-features = false, features = ["std", "wgsl", "dx12", "metal", "vulkan",
"gles"]` (`Cargo.toml:21`). There is no `webgpu` or `webgl` feature, so there is no browser backend.

A second device path exists for tests: `Offscreen::new` (`gpu.rs:510-516`) uses
`InstanceDescriptor::new_without_display_handle_from_env()`, `RequestAdapterOptions::default()` and
`DeviceDescriptor::default()`, and returns `None` on a machine without an adapter.

## 2. Swapchain, resize, surface-lost, device-lost

- **Resize is deferred.** `WindowEvent::Resized` calls `Renderer::resize(w, h)` (`app.rs:676-680`),
  which only stores the size and sets `resized = true` (`gpu.rs:219-225`). The surface is configured
  once at the start of the next `render` (`gpu.rs:239-241`), "however many sizes passed by", because
  configuring waits for the GPU.
- **Zero size means hidden.** `resize` sets `hidden = width == 0 || height == 0`; `render` then returns
  `Frame::Skipped` and never configures a zero-sized surface (`gpu.rs:220`, `236-238`).
- **`get_current_texture` outcomes** (`gpu.rs:243-260`), reported to the app as
  `enum Frame { Drawn, Retry, Skipped }` (`gpu.rs:87-96`):

  | wgpu result | Action |
  |---|---|
  | `Success(frame)` | draw |
  | `Suboptimal(frame)` | reconfigure, then draw this frame anyway |
  | `Outdated`, `Lost` | reconfigure; return `Retry` the first time, `Skipped` if the previous frame also retried (`retried` flag, so a refusing surface cannot spin the loop) |
  | `Timeout`, `Occluded`, `Validation` | return `Skipped` |

  On `Retry` the app calls `window.request_redraw()` at once (`app.rs:1553-1556`). On `Skipped` while
  something animates it waits `SKIPPED_RETRY` = 100 ms (`app.rs:61`, `1599-1600`). The surface is never
  re-created on `Lost`, only reconfigured.
- **Device lost.** `Renderer::is_lost()` reads the flag (`gpu.rs:206`). `App::frame` checks it first and
  calls `App::recover_gpu` (`app.rs:1460-1478`, `1482-1484`), which builds a whole new `Renderer`
  (instance, surface, adapter, device), clears the glyph atlas and replaces the image atlas (their
  texels lived on the old device), and re-applies the window size. If that fails it logs and retries
  every `GPU_RETRY` = 2 s (`app.rs:58`); the retry time is one of the `ControlFlow::WaitUntil`
  candidates (`app.rs:1636`).
- **Presentation** is `self.queue.submit([commands]); self.queue.present(frame);` (`gpu.rs:264-265`).
  In wgpu 30 `present` is a method of `Queue`, not of `SurfaceTexture`.
- **Error scopes:** not used anywhere (no `push_error_scope` in the crate).

## 3. Frame pacing

The rule (`CLAUDE.md`, "Beautiful, elegant and fast"): redraw only when something changes; an idle app
draws nothing.

- **Event loop.** `EventLoop::<WorkerEvent>::with_user_event().build()` and `event_loop.run_app(&mut
  handler)` (`main.rs:158-163`). `Handler::about_to_wait` sets the control flow from `App::control_flow`
  on every iteration (`main.rs:86-90`).
- **`ControlFlow`** (`app.rs:1621-1640`): `ControlFlow::Wait` when nothing is due, else
  `ControlFlow::WaitUntil(min(..))` over: the next caret blink flip (only when the window is seen and
  focused), a pending tooltip, the GPU retry time, `redraw_at`, and the shell's next wall-clock deadline
  (clamped to 1-60 s). `ControlFlow::Poll` is never used.
- **Timer wake.** `Handler::new_events` calls `app.wake()` on `StartCause::ResumeTimeReached`
  (`main.rs:55-59`), which ticks the shell and requests a redraw (`app.rs:1643-1648`).
- **Redraw on change.** Every handled window event ends with `self.window.request_redraw()`
  (`app.rs:774`), as does every worker event (`app.rs:934`). Drawing happens only in
  `WindowEvent::RedrawRequested` (`app.rs:669-675`), with one exception: the very first frame is drawn
  directly from `App::new` (`app.rs:603-606`) because a hidden window never receives `RedrawRequested`
  on Windows. The window is created with `.with_visible(false)` and shown after that first frame
  (`app.rs:483`, `1557-1567`).
- **Animation ticking.** There is no animation scheduler. A widget that is mid-animation sets
  `ui.animating = true` during the frame (`ui.rs:162-173`); at the end of `App::frame`, if it is set,
  the app calls `window.request_redraw()` again (`app.rs:1597-1606`). Frame rate is then whatever vsync
  gives. `dt` is measured per frame and clamped to 0.05 s (`app.rs:1492`).
- **Background activity.** While the app is "busy" but nothing eases, it draws at about 60 fps on whole
  refreshes of the monitor: `busy_interval(refresh_millihertz)` returns `floor(hz / 60)` refresh
  periods, or `None` (draw every refresh) when that is 1 or the rate is unknown (`app.rs:63`,
  `1839-1843`). The refresh rate comes from `window.current_monitor().refresh_rate_millihertz()`
  (`app.rs:1832`), re-read on `Resized` and `Moved`.
- **Reduced motion.** `platform::reduce_motion()` is read on a worker at start and whenever the window
  regains focus (`app.rs:602`, `693`); it sets `ui.still`, and then busy indicators hold still and the
  window is redrawn once a second (`app.rs:1607-1610`). `Ui::motion()` returns `None` in that mode
  (`ui.rs:150-152`).
- **Unseen windows are not drawn.** `App::unseen()` is true when hidden to the tray, occluded
  (`WindowEvent::Occluded`), or zero-sized (`app.rs:1615-1617`); `frame` returns early.
- **Vsync:** `PresentMode::AutoVsync`, no frame limiter of its own beyond the above.

## 4. Shaders

- One shader file for the whole application: `crates/desktop/src/shader.wgsl` (139 lines), WGSL only.
- Loaded at compile time: `wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())` (`gpu.rs:274-277`).
- **Hot reload: no equivalent.** No file watching, no runtime shader loading, no preprocessor, no
  includes, no shader variants.
- A unit test parses and validates it with naga: `naga::front::wgsl::parse_str` then
  `naga::valid::Validator::new(ValidationFlags::all(), Capabilities::empty())` (`gpu.rs:570-582`),
  reached as `wgpu::naga` (no separate naga dependency).
- Entry points are named `vs_main` and `fs_main`. The file opens with a comment block documenting every
  quad kind and the units.
- Rule in `CLAUDE.md`: "New visual effects go into that pipeline as new quad kinds, never as extra passes."

## 5. Pipeline, bind group, vertex layout, quad kinds, blending, draw calls

One `RenderPipeline`, one `BindGroup`, one instance buffer, **one draw call per frame**
(`pass.draw(0..4, 0..instances.len() as u32)`, `gpu.rs:487`). Built in `Draw::new` (`gpu.rs:273-401`).

**Bind group 0** (the only group; label `"ui"`):

| Binding | Type | Visibility | Content |
|---|---|---|---|
| 0 | uniform buffer, `min_binding_size: None`, no dynamic offset | `VERTEX_FRAGMENT` | `Globals` |
| 1 | `texture_2d<f32>`, filterable | `FRAGMENT` | glyph atlas, `R8Unorm` |
| 2 | sampler, `Filtering` | `FRAGMENT` | the one sampler |
| 3 | `texture_2d<f32>`, filterable | `FRAGMENT` | image atlas, `Rgba8Unorm` |

**Geometry.** No vertex buffer of vertices. The only vertex buffer is per-instance
(`step_mode: wgpu::VertexStepMode::Instance`), and the four corners come from `@builtin(vertex_index)`:
`corner = vec2(f32(vertex & 1u), f32(vertex >> 1u))` (`shader.wgsl:51`). Topology `TriangleStrip`.

**Instance layout** (`gpu.rs:13-28`; 6 x `Float32x4` = 96 bytes; locations 0-5 via
`wgpu::vertex_attr_array![0 => Float32x4, ...]`, `gpu.rs:368`):

| Loc | Field | Meaning |
|---|---|---|
| 0 | `rect: [f32; 4]` | x, y, w, h in **logical** pixels |
| 1 | `clip: [f32; 4]` | x0, y0, x1, y1 in logical pixels |
| 2 | `color: [f32; 4]` | fill or glyph colour, straight-alpha sRGB |
| 3 | `color2: [f32; 4]` | border colour, or gradient bottom colour |
| 4 | `uv: [f32; 4]` | atlas rectangle in **texels**: x, y, w, h |
| 5 | `params: [f32; 4]` | corner radius, border width, blur radius, **kind** |

**Quad kinds** are a float in `params.w` (`shader.wgsl:1-14`, constants in `paint.rs:88-92`):

| Kind | Name | Fragment behaviour |
|---|---|---|
| 0.0 | `KIND_SHAPE` | Rounded-rectangle SDF. With `params.y > 0`: inner border in `color2`. With `params.z > 0`: soft shadow, alpha `1 - smoothstep(-blur, blur, d)` |
| 1.0 | `KIND_GLYPH` | Alpha = `color.a` x coverage from the R8 atlas |
| 2.0 | `KIND_IMAGE` | RGBA from the image atlas, clipped to a rounded rectangle, faded by `color.a` |
| 3.0 | `KIND_GRADIENT` | Rounded rectangle, vertical `smoothstep` gradient from `color` to `color2` |

**Vertex stage** (`shader.wgsl:49-79`): multiplies logical geometry by `globals.scale`; for non-glyph
kinds it rounds origin and size to whole physical pixels (crisp 1 px borders) and pads the quad by
`blur * scale + 1` for anti-aliasing and blur falloff. Glyph quads are not rounded there (the CPU
already placed them on the pixel grid). All per-instance data is passed `@interpolate(flat)`.

**Fragment stage** (`shader.wgsl:94-139`): clip test against `in.position.xy` with `discard`; edge
anti-aliasing is `clamp(0.5 - d, 0.0, 1.0)` on the SDF; output is premultiplied
(`vec4(color.rgb * color.a, color.a)`).

**Blending:** `wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING`, `write_mask: ColorWrites::ALL`
(`gpu.rs:389-393`). `depth_stencil: None`, `multisample: MultisampleState::default()` (no MSAA),
`multiview_mask: None`, `cache: None`.

**Ordering.** Painter's algorithm: instances are drawn in the order the UI code pushed them. Overlays
(menus, modals, tooltips) are drawn last (`shell/mod.rs:1443-1452`).

**Instance buffer.** Starts at 1024 instances; when too small it is re-created at
`instances.len().next_power_of_two()`; filled each frame with `queue.write_buffer` (`gpu.rs:399-410`,
`455-461`). The CPU-side `Vec<Instance>` is owned by `App` and cleared by `Painter::new`
(`paint.rs:120`).

**Labels.** Every wgpu object gets a short lower-case label: `"ui"`, `"globals"`, `"instances"`,
`"glyph atlas"`, `"image atlas"`, `"atlas"`, `"snapshot"`.

## 6. Uniform and storage buffers; POD casting

- One uniform struct, mirrored field for field in Rust and WGSL (`gpu.rs:35-44`, `shader.wgsl:16-22`),
  32 bytes, with explicit trailing padding:

  ```rust
  // crates/desktop/src/gpu.rs
  #[repr(C)]
  #[derive(Clone, Copy)]
  struct Globals {
      viewport: [f32; 2],     // physical pixels
      scale: f32,             // physical pixels per logical pixel
      linear_output: f32,     // 1.0 when the surface format is sRGB-encoded
      atlas_size: [f32; 2],
      pad: [f32; 2],
  }
  ```

- Flags are passed as `f32` (0.0 or 1.0), never `u32` or `bool`.
- **bytemuck** is used without its `derive` feature. `Pod` and `Zeroable` are implemented by hand with
  a `// SAFETY:` comment each (`gpu.rs:30-33`, `46-49`):

  ```rust
  // SAFETY: `#[repr(C)]`, only `f32` fields, no padding, and all-zero is valid.
  unsafe impl bytemuck::Zeroable for Instance {}
  // SAFETY: see above; every bit pattern is a valid `f32`.
  unsafe impl bytemuck::Pod for Instance {}
  ```

- Casting: `bytemuck::bytes_of(&globals)` for one struct, `bytemuck::cast_slice::<Instance, u8>(instances)`
  for slices (`gpu.rs:453`, `455`). Uploads always go through `queue.write_buffer(&buf, 0, bytes)`;
  no mapped-at-creation buffers, no staging belt.
- Buffer sizes are written `size_of::<T>() as u64` (`size_of` from the Rust 2024 prelude).
- **Storage buffers: no equivalent** (and the requested limits forbid them, see section 1).
- **Dynamic offsets, multiple bind groups, push constants / immediates: no equivalent**
  (`immediate_size: 0`).

## 7. Textures and samplers

| Item | Convention | Evidence |
|---|---|---|
| Glyph atlas | `R8Unorm`, 2048 x 2048 (`ATLAS_SIZE`, "the largest size every backend (including GLES) guarantees"), 1 mip, usage `TEXTURE_BINDING \| COPY_DST` | `text.rs:19-21`, `gpu.rs:285-298` |
| Image atlas | `Rgba8Unorm` (deliberately **not** `Rgba8UnormSrgb`: "texels are sRGB values like every other colour here"), same size and usage | `gpu.rs:299-301` |
| Sampler | One, label `"atlas"`: `mag_filter` and `min_filter` `Linear`, everything else default (clamp to edge, nearest mip). Glyph quads map texels 1:1 so linear "only smooths float rounding error" | `gpu.rs:302-309` |
| Sampling | `textureSampleLevel(tex, atlas_sampler, uv, 0.0)`; UVs are computed in the vertex shader from texel rectangles divided by `globals.atlas_size` | `shader.wgsl:72`, `104`, `107` |
| Mipmaps | None | |
| Upload | CPU code queues `Upload { x, y, w, h, data }` rectangles; `Draw::upload` drains them with `queue.write_texture` before the frame is acquired (1 byte per texel for glyphs, 4 for images) | `text.rs:510-523`, `gpu.rs:413-433` |
| Padding | Each atlas item is stored with one transparent texel column on the right and row at the bottom (`pad()`), and the padding is uploaded too | `text.rs:596-632` |
| Packing | Shelf packer with LRU eviction shared by both atlases: shelf heights rounded up to 8 texels (`BUCKET`), the least recently used shelf is emptied when full, never one used in the current frame | `atlas.rs` (whole file) |

**sRGB handling.** Every colour in the application is a straight-alpha sRGB `[f32; 4]` (`paint.rs:9-10`).
Blending happens on sRGB-encoded values, on purpose, by choosing a non-sRGB surface format. Only if the
surface offers nothing but sRGB formats does the shader convert at the very end
(`globals.linear_output > 0.5` then `srgb_to_linear(color.rgb)`, `shader.wgsl:135-137`), and the clear
colour is converted on the CPU (`gpu.rs:473-477`, `566-568`). There is no linear-light pipeline.

**Images** (`image.rs`, `paint.rs:126-210`): a file is decoded on a worker thread, scaled on the CPU to
the exact physical pixel size it will be drawn at (cover-and-crop, or fit for a "whole" picture), EXIF
orientation applied, and inserted into the image atlas as RGBA8. The atlas key is
`ImageKey { path, w, h, whole }`. Decoded pixels are not kept on the CPU: an evicted image is decoded
again. Limits: 32 MiB file, 1024 px thumbnail side, 2047 x 1536 whole picture. At most one decode worker
per frame (`app.rs:1583-1596`). Decoders live in `atroxabot_core::picture` (crates `png`, `zune-jpeg`,
`image-webp`). A blurred backdrop is a 24-texel thumbnail stretched over a rectangle so the sampler
does the blur (`paint.rs:194-210`).

## 8. Render graph or pass abstraction

**No render graph. No equivalent.** There is one render pass per frame (label `"ui"`, clear then
store, one colour attachment, no depth). The only structuring is a private `struct Draw`
(`gpu.rs:124-133`) with three methods, shared by the window renderer and the test-only `Offscreen`:

- `Draw::new(device, format)` builds the pipeline, atlases, sampler, bind group and buffers.
- `Draw::upload(queue, glyphs, images)` copies pending atlas rectangles.
- `Draw::encode(device, queue, (view, size), instances, scale, clear) -> wgpu::CommandBuffer` writes the
  uniforms and instances and records the pass into any target view.

`Renderer` (`gpu.rs:103-120`) owns surface, device, queue, config and a `Draw`; `Renderer::render` is
the only per-frame entry point and returns `Frame`.

## 9. Math types

**No math crate and no math module. No equivalent for vectors, matrices or quaternions.** The whole
application uses:

- `pub type Color = [f32; 4];` (straight-alpha sRGB) with `hex(0xRRGGBB)`, `hexa(rgb, alpha)`,
  `fade(color, factor)`, `mix(a, b, t)`, all in `paint.rs:9-34`.
- `pub struct Rect { x, y, w, h: f32 }` in logical pixels with `new`, `right`, `bottom`, `contains`,
  `intersect` (`paint.rs:36-86`).
- Tuples `(f32, f32)` for points and sizes, arrays `[f32; 4]` for GPU data.
- Colour-space helpers written inline where needed: OKLCH to sRGB (`theme.rs:317-340`), HSV both ways
  (`color.rs:284-307`), sRGB to linear (`gpu.rs:566`).

All `f32`. Units are logical pixels at the API, physical pixels inside text layout and the shader.

## 10. Errors and logging

**Errors** (rule in `CLAUDE.md`: no `thiserror`, no `anyhow`; "write error types by hand: `Display` +
`std::error::Error`"):

- One `#[derive(Debug)] pub enum` per concern, each variant documented, with a hand-written
  `impl fmt::Display` (messages are lower-case sentences for the user, no trailing full stop) and
  `impl std::error::Error` with `source()`. Examples: `GpuError` (`gpu.rs:51-84`), `StartupError`
  (`app.rs:350-375`), `atroxabot_core::Error` (`crates/core/src/error.rs`), the MCP crate's
  `pub(crate) enum Error` (`crates/mcp/src/error.rs`).
- `pub type Result<T, E = Error> = std::result::Result<T, E>;` in the core crate (`error.rs:6-7`).
- Conversion is `map_err(Variant)` at the call site (`gpu.rs:145`, `153`, `162`) or a hand-written
  `impl From<..>` (`error.rs:224-232`).
- Error types carry predicate methods (`is_retryable`, `is_unauthorized`, `ends_connection`) rather
  than callers matching on variants.
- "No `unwrap`s on fallible runtime paths." `expect("...")` appears only for build defects (embedded
  fonts, `font.rs:365`) and in tests.
- Background failures reach the user as a `String` posted to the UI thread (`WorkerEvent::Problem`,
  `app.rs:172`), shown as a dismissible notice.
- Fatal start-up failures show a native dialog (`platform::alert`) and return `ExitCode::FAILURE`
  (`main.rs:112-116`, `158-170`). `main` returns `ExitCode`, not `Result`.

**Logging** (`crates/desktop/src/log.rs`, 108 lines; no `log`, `tracing` or `env_logger` crate):

- API: `log::init()` (called first in `main`) and `log::error(message: impl Into<String>)`. **One
  level only.** There is no info, debug or trace.
- `error` prints `eprintln!("atroxabot: {message}")` and sends `"{unix_seconds} {message}"` over an
  `mpsc` channel to a dedicated thread (`"atroxabot-log"`) that appends to `~/.atroxabot/desktop.log`.
  The sender lives in a `static LINES: OnceLock<Sender<String>>`.
- Rotation: at start, a log over 1 MiB (`MAX_LOG = 1 << 20`) is renamed to `desktop.log.old`.
- Files are created `0600` and the directory `0700` on Unix.
- Panic hook (`install_panic_hook`): calls the default hook, writes `desktop-crash.log` with version,
  thread name, panic info and `std::backtrace::Backtrace::force_capture()`, then shows a native alert.
  Needed because release builds use `panic = "abort"` and have no console on Windows
  (`#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`, `main.rs:9`).
- wgpu validation errors are routed into this log (section 1). wgpu's own `log` output is not captured.

## 11. Dependencies, profiles, lints

Workspace: `resolver = "3"`, `edition = "2024"`, `rust-version = "1.89"`, version shared through
`[workspace.package]`. No `rust-toolchain.toml` (CI uses stable). Six crates: `claude-agent`, `codex`,
`mcp`, `core`, `voice`, `desktop`. Dependency policy (`CLAUDE.md`): "Add one only if it is truly required
and can't reasonably be written by hand, with a `# why:` comment ... no async runtimes (threads and
blocking I/O), no convenience crates."

**All third-party dependencies of the workspace** (`Cargo.toml:12-37`; versions as locked in `Cargo.lock`):

| Crate | Version | Features | Used for | Used by |
|---|---|---|---|---|
| `winit` | 0.30.13 | default | window, event loop, input | desktop |
| `wgpu` | 30.0.1 | `default-features = false`; `std`, `wgsl`, `dx12`, `metal`, `vulkan`, `gles` | rendering | desktop |
| `ttf-parser` | 0.25.1 | no default; `std`, `opentype-layout` | font parsing, GPOS kerning, GSUB ligatures, COLR layers, glyph outlines | desktop |
| `bytemuck` | 1.25.2 | default (no `derive`) | casting instance and uniform data to bytes | desktop |
| `arboard` | 3.7.0 | no default | clipboard text; on Windows also the copied-file list | desktop |
| `png` | 0.18.1 | default | PNG decode and encode (snapshots, pasted pictures, window icon) | desktop, core |
| `zune-jpeg` | 0.5.15 | no default; `std`, `x86`, `neon` | JPEG decode | core |
| `image-webp` | 0.2.4 | default | WebP decode | core |
| `serde` | 1.0.229 | `derive` | saved data | core |
| `serde_json` | 1.0.154 | default | JSON | all |
| `ureq` | 3.5.1 | no default; `rustls` | blocking HTTPS | core, mcp, voice |
| `flate2` | 1.1.11 | default | gzip of downloaded archives | core |
| `ring` | 0.17.14 | default | Ed25519 and SHA-256 for updates | core |
| `cpal` | 0.18.2 | default | audio capture and playback | voice |

Direct dependencies of the desktop crate (`crates/desktop/Cargo.toml`): `atroxabot-core`,
`atroxabot-voice`, `winit`, `wgpu`, `ttf-parser`, `bytemuck`, `arboard`, `png`, `serde_json`. No
build-dependencies: `build.rs` uses only `std`.

Notably **absent**: any UI framework, any math crate, `pollster`/`futures`, `log`/`tracing`,
`thiserror`/`anyhow`, `rfd`, `raw-window-handle` as a direct dependency, `image`, `fontdue`/`cosmic-text`/
`rustybuzz`, `windows`/`windows-sys` (Win32 is called through hand-written `unsafe extern "system"`
blocks), `tokio`.

**Profiles** (`Cargo.toml:58-66`):

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

# Font rasterisation and shader compilation are painfully slow unoptimised.
[profile.dev.package."*"]
opt-level = 3
```

**Lints** (`Cargo.toml:39-56`, inherited with `[lints] workspace = true` in every crate):

```toml
[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"
missing_docs = "warn"

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
cast_possible_truncation = "allow"
cast_precision_loss = "allow"
cast_sign_loss = "allow"
cast_possible_wrap = "allow"
module_name_repetitions = "allow"
too_many_lines = "allow"
unreadable_literal = "allow"
struct_excessive_bools = "allow"
assert_is_empty = "allow"
```

CI runs `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` and
`cargo test --workspace --locked` on `ubuntu-latest`, `windows-latest`, `macos-latest`
(`.github/workflows/ci.yml`). So warnings are errors in practice.

`rustfmt.toml`: `use_small_heuristics = "Max"`, `max_width = 120`. `clippy.toml`: only
`doc-valid-idents`. `.gitattributes`: `*.sh text eol=lf`.

## 12. Web / native split

**No equivalent. Native only** (Windows, macOS, Linux). There is no `wasm32` target, no
`cfg(target_arch = "wasm32")`, no `webgpu`/`webgl` wgpu feature, no `web-sys`. Platform differences are
handled inline with `cfg!(target_os = "...")` expressions and `#[cfg(windows)]` /
`#[cfg(target_os = "macos")]` / `#[cfg(all(unix, not(target_os = "macos")))]` blocks inside ordinary
functions (`platform.rs`, `app.rs:486-500`, `2014-2034`), not with per-platform modules, except
`tray.rs`, which has a Windows module and a no-op stand-in.

## 13. Input and windowing

**Structure.** `main.rs` holds a small `Handler` implementing `ApplicationHandler<WorkerEvent>` that
forwards to `App` (created in `resumed`, because a window needs a running loop). `App::window_event`
(`app.rs:664-775`) translates winit events into fields of a plain `Ui` struct (`ui.rs:17-67`) and into
calls on the shell. Background threads talk to the UI thread only through
`EventLoopProxy<WorkerEvent>::send_event` (`app.rs:635-644`). The winit thread owns all state.

| Concern | Convention | Evidence |
|---|---|---|
| Window attributes | `Window::default_attributes().with_title(..).with_inner_size(LogicalSize).with_min_inner_size(..).with_maximized(..).with_theme(Some(..)).with_visible(false).with_decorations(!titlebar::CUSTOM)` | `app.rs:474-485` |
| Window size | First start 1240 x 820 logical, minimum 820 x 560 (scaled by zoom, capped at 90 % of the monitor); size and maximized state saved on exit as `"1200x800 maximized"` | `app.rs:50-52`, `1848-1868`, `1668-1685` |
| Decorations | Custom title bar everywhere except macOS (`titlebar::CUSTOM = !cfg!(target_os = "macos")`). Windows: `WindowAttributesExtWindows::with_undecorated_shadow(attributes, true)`. macOS: `.with_titlebar_transparent(true).with_fullsize_content_view(true).with_title_hidden(true)`. Linux: `WindowAttributesExtWayland::with_name(attributes, "atroxabot", "atroxabot")` | `app.rs:486-500`, `titlebar.rs:14` |
| Title bar | `titlebar::grab(view, mouse, maximized)` decides per press: 5 px edge zone gives `Grab::Resize(ResizeDirection)` then `window.drag_resize_window(direction)`; the 36 px top strip gives `Grab::Move` then `window.drag_window()`; a second press within 0.4 s toggles `set_maximized`. Three 46 px wide buttons (minimize, maximize, close) are drawn by `titlebar::draw` with the same pipeline | `titlebar.rs`, `app.rs:704-727` |
| Mouse position | `CursorMoved`: physical position divided by `scale` (display scale x zoom) into logical `ui.mouse`. `CursorLeft`: `(f32::MIN, f32::MIN)` | `app.rs:700-703` |
| Mouse buttons | **Left button only.** Press calls `Ui::press(now)` (sets `down`, `pressed`, `press_pos`, counts click bursts: within 0.4 s and 4 px); release sets `down = false`, `released = true`. No right or middle button, no touch events | `app.rs:704-727`, `ui.rs:109-116` |
| Scroll | `MouseScrollDelta::LineDelta(x, y)` is `x * 100.0, y * 100.0` logical px; `PixelDelta(p)` is `p / scale`. Shift turns a vertical wheel sideways. Accumulated into `ui.scroll` / `ui.scroll_x` (positive = down / right), consumed and zeroed each frame | `app.rs:728-737`, `ui.rs:119-123` |
| Modifiers | `WindowEvent::ModifiersChanged(mods) => self.ui.mods = mods.state()`. "Primary" is Ctrl, or Super on macOS; "word" is Ctrl, or Alt on macOS | `app.rs:699`, `ui.rs:477-480` |
| Keyboard | Only `KeyboardInput { event, is_synthetic: false, .. }` with `event.state == ElementState::Pressed` (repeats included, releases ignored). Matching is on `event.logical_key` (`Key::Named(NamedKey::..)`, `Key::Character(c)` compared case-insensitively). Physical key codes are never used | `app.rs:738-761` |
| Text input | Typed text comes from `event.text` of the same key event, inserted when the primary modifier is not held, or when Alt is also held (AltGr on Windows/Linux) | `ui.rs:502-506` |
| IME | `window.set_ime_allowed(true)` at start. `Ime::Commit(text)` inserts; `Ime::Preedit(text, cursor)` is shown inline at the caret, underlined 1.5 px in the accent colour; `Ime::Disabled` clears it. The candidate window follows the caret via `window.set_ime_cursor_area(LogicalPosition, LogicalSize)` (values multiplied by zoom), re-sent only when the caret rectangle changes | `app.rs:504`, `762-770`, `1572-1581`, `shell/composer.rs:280-372` |
| Text editing | `Editor` (`editor.rs`): string + caret + anchor as byte offsets, optional char limit and filter, word and line movement, a practical subset of grapheme clustering. `ui::edit_key` maps keys (arrows, Home/End, Backspace/Delete, select all, copy, cut, paste) | `editor.rs`, `ui.rs:472-509` |
| Cursor icons | `ui.cursor` is reset to `CursorIcon::Default` in `Ui::begin`; hovered widgets overwrite it; after the frame the app calls `window.set_cursor(icon)` only if it changed. Icons used: `Default`, `Pointer`, `Text`, `Crosshair`, `Grabbing`, `NsResize`, `EwResize`, `NwseResize`, `NeswResize` | `ui.rs:94`, `app.rs:1568-1571` |
| DPI | `WindowEvent::ScaleFactorChanged` is **not** handled. `App::scale()` reads `window.scale_factor() as f32 * zoom / 100` whenever needed; `App::view()` is `inner_size / scale`. UI code works in logical pixels; the shader multiplies by `globals.scale`; text is laid out in physical pixels | `app.rs:619-628` |
| Zoom | Primary + `+`/`=`, `-`, `0`: 50-200 % in steps of 10, saved in the config, also scales the minimum window size | `app.rs:54-56`, `746-758`, `1815-1824` |
| Focus | `WindowEvent::Focused(f)` sets `ui.focused` (the caret blinks only when focused) | `app.rs:688-696` |
| File drop | `WindowEvent::DroppedFile(path)` | `app.rs:698` |
| Window theme | `window.set_theme(Some(Theme::Light or Theme::Dark))` follows the colour scheme | `app.rs:614`, `2005-2007` |
| Window icon | Windows: `Icon::from_resource(1, Some(PhysicalSize))` from the `.res` that `build.rs` writes by hand, plus `set_taskbar_icon`. X11: `Icon::from_rgba` of an embedded 256 px PNG. macOS and Wayland: none | `app.rs:2014-2034`, `build.rs` |
| Attention | `window.request_user_attention(Some(UserAttentionType::Informational))` when unfocused | `app.rs:1209-1211` |
| Clipboard | `arboard::Clipboard` held as `Option<Clipboard>` on `App`, created lazily on first use; `set_text` / `get_text`, failures ignored. Non-text paste (`platform::clipboard_files`): Windows asks `clipboard.get().file_list()` then Win32 (`OpenClipboard`, `GetClipboardData` for the registered `"PNG"` format or `CF_DIB`) by hand-written FFI; macOS `osascript`; Linux `wl-paste` then `xclip` | `ui.rs:441-454`, `platform.rs:230-435` |
| File dialogs | No dialog crate. `platform::pick_files`, `pick_any`, `pick_image`, `pick_folder` run a helper process and parse stdout lines: PowerShell (`pwsh` then `powershell`) with `System.Windows.Forms` on Windows, `osascript` on macOS, `zenity` then `kdialog` on Linux. They block, so they are always called from a worker thread. User text is passed as environment variables or arguments, never spliced into a script. **Open only: there is no save dialog** | `platform.rs:184-539`, `app.rs:1113-1139` |
| Other shims | `open_url` (`rundll32 url.dll,FileProtocolHandler` / `open` / `xdg-open`), `open_folder`, `alert` (Win32 `MessageBoxW` / `osascript` / `zenity`), `reduce_motion`, `run_in_terminal`, `no_window` (`CREATE_NO_WINDOW`) | `platform.rs` |
| Multi-window | **No equivalent.** One window; the `WindowId` argument is ignored (`main.rs:80`). A Windows tray icon exists (`tray.rs`, Win32 FFI) | |
| Single instance | Lock file `~/.atroxabot/desktop.lock`; a Windows named mutex for the installer | `main.rs:108-152`, `platform.rs:73-88` |
| Exit | `event_loop.exit()`; state is saved in `impl Drop for App`, and `Handler::exiting` drops the app explicitly because on macOS `run_app` may never return | `main.rs:92-105`, `app.rs:1651-1687` |

## 14. Text rendering

Entirely hand-written on top of `ttf-parser`; files `font.rs`, `text.rs`, `raster.rs`, `atlas.rs`.

**Fonts** (`font.rs`). Nine faces embedded with `include_bytes!("../assets/...")` and parsed once in
`Fonts::load()` (`font.rs:364-381`), identified by `FontId(u8)` constants:

| `FontId` | File | Use |
|---|---|---|
| `REGULAR` (0) | `Inter-Regular.ttf` | body and UI text |
| `SEMIBOLD` (1) | `Inter-SemiBold.ttf` | titles, labels, emphasis |
| `ITALIC` (2) | `Inter-Italic.ttf` | |
| `MONO` (3) | `JetBrainsMono-Regular.ttf` | code |
| `EMOJI` (4) | `Twemoji.Mozilla.ttf` | colour emoji (COLR) |
| `SERIF` (5), `SERIF_ITALIC` (6), `SERIF_BOLD` (7) | `Literata-{Regular,Italic,SemiBold}.ttf` | prose |
| `ICON` (8) | `Lucide.ttf` | icons |

There are only two weights (regular and semibold); no variable fonts, no synthetic bold or italic.

**Fallback** (`Fonts::resolve`, `font.rs:396-423`): requested face, then the emoji face, then Inter
Regular, then **system fonts by hard-coded path lists** per OS (`system_fallbacks`, `font.rs:273-336`;
for example `%FONTS%/msyh.ttc#0`, `/System/Library/Fonts/PingFang.ttc#0`,
`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc#0`), each guarded by a coverage predicate
(`is_cjk`, `is_hangul`, `not_cjk`) and loaded lazily on first need (`OnceCell`, data leaked with
`Box::leak`). A following U+FE0F forces the emoji face. Results are cached per `(FontId, char)`. A
character nothing covers becomes glyph 0 of the requested face.

**Shaping: none.** One `char` maps to one glyph, left to right. Documented limitation
(`font.rs:14-15`): no Arabic joining, Indic reordering or bidi. What exists:

- Pair kerning from GPOS `kern` lookups (pair adjustment formats 1 and 2), falling back to the `kern`
  table, cached per face in a `RefCell<HashMap>` (`font.rs:108-151`). Applied only between glyphs of
  the same face and size.
- Emoji sequences (ZWJ, flags, skin tones, keycaps) are collected into a cluster and collapsed with the
  emoji font's GSUB ligature lookups (`ccmp`, `liga`, `rlig`, `clig`), longest match
  (`text.rs:295-313`, `font.rs:155-181`). No ligatures for text faces.
- Colour emoji: COLRv0 only. `Face::layers` returns `(glyph id, Option<rgba>)` layers; each layer is
  rasterised as a normal coverage glyph and drawn as its own quad in a fixed colour
  (`font.rs:184-216`, `text.rs:705-712`). Gradient (COLRv1) paints are skipped.

**Rasterisation** (`raster.rs`): **coverage bitmaps, not SDF.** Outlines from
`ttf_parser::Face::outline_glyph` are flattened to line segments (tolerance 0.1 px; at most 64
segments per curve) and accumulated as exact signed areas, then integrated per row (the `font-rs`
technique) into 8-bit coverage, non-zero winding. No supersampling, no hinting, no LCD subpixel
anti-aliasing, no gamma correction of coverage. `rasterize_outline` does the same for arbitrary
`OutlineBuilder` callers (used for logos drawn from SVG path data, `logo.rs`).

**Atlas** (`text.rs:525-620`): `GlyphAtlas` caches one bitmap per `(FontId, glyph index,
px.to_bits())`, that is, **per exact pixel size**; scaling or zooming rasterises again. Packed by the
shelf allocator with one texel of padding. When full, the least recently used shelf is evicted; if
even that fails (one frame needs more than the atlas), the painter sets `atlas_full`, the app clears
the atlas and **draws the same frame a second time** (`app.rs:1518-1537`).

**Subpixel positioning: none.** The layout origin and every glyph pen position are rounded to whole
physical pixels at draw time (`text.rs:662-663`, `683`); advances and kerning stay fractional in the
layout. The line height and the baseline are rounded to whole physical pixels.

**Layout** (`TextLayout`, `text.rs:115-482`):

- Built by `TextLayout::new(fonts, text, style, max_width, scale)` or `TextLayout::rich(..)` with
  styled runs (`Run { font, size (relative), ink (colour slot), decoration, link }`). Internally in
  physical pixels, public API in logical pixels.
- `Style { font: FontId, size: f32 (logical px), line_height: f32 (multiple of size) }`.
- Vertical metrics: `line_height = round(size * line_height * scale)`;
  `baseline = round(midpoint(line_height, cap_height_px))`, so capitals sit in the middle of the line
  (`text.rs:244-251`). Ascender and descender are not used.
- Wrapping: greedy; break opportunities after whitespace and around CJK characters; a word wider than
  the line breaks anywhere; `\n` is a hard break; a tab advances four spaces; zero-width format
  characters are dropped. Alignment: `Align::Left` or `Align::Center` only.
- Queries: `width`, `height`, `line_height`, `baseline`, `line_count`, `truncate` (adds an ellipsis),
  `caret(byte) -> (x, y)`, `hit(x, y) -> byte`, `selection_spans`, `line_spans`, `link_at`,
  `decorations`.
- `TextLayout::place(..)` culls lines and glyphs against the visible rectangle and emits `PlacedGlyph`
  quads; `Painter::glyphs` turns them into `KIND_GLYPH` instances.
- Decorations (inline-code background, underline, strike-through) are drawn as extra rectangles by
  `Painter::rich` (`paint.rs:292-308`).

**Layout caching.** Ordinary labels are **laid out again every frame** (`Painter::label`,
`paint.rs:360-364`); there is no global layout cache. Only long documents are cached: each
conversation block keeps a `Doc` in a `HashMap<(usize, usize), Cached>`, invalidated when the text
length, the width or the scale changes (`Doc::fits`), with a budget of 4 ms per frame for re-laying
out off-screen blocks during a resize (`shell/thread.rs:48-52`, `216-262`, `doc.rs:437-441`).

**Icons** (`ui.rs:261-382`): the Lucide icon font, one private-use code point per icon, as `pub const`
`char`s in `ui::icon` (for example `CHECK = '\u{e06c}'`, `X = '\u{e1b2}'`, `SEARCH = '\u{e151}'`).
`icon::draw(p, glyph, (cx, cy), size, color)` lays the glyph out with `theme::icon(size)`
(`line_height: 1.0`) and positions it at `x = cx - width / 2`, `y = cy - size * 0.35`, both rounded. A
test checks that every listed code point resolves in the icon font and has an advance of one em.
Brand marks are separate: SVG path data rasterised through the same rasteriser into the glyph atlas
(`Painter::logo`, `paint.rs:334-357`).

## 15. UI architecture

**Immediate mode, hand-written, no retained widget tree** (`main.rs:3-6`; rule in `CLAUDE.md`: "No UI
frameworks (egui, iced, Tauri, ...)").

**Per frame** (`App::frame`, `app.rs:1481-1611`):

1. `ui.begin(dt, time)` resets the cursor, `animating`, tooltip state, and prunes animation values not
   used in the previous frame.
2. `Painter::new(&fonts, (&mut atlas, &mut images), &mut instances, scale, view, &palette)` clears the
   instance list and starts an atlas frame.
3. `shell.draw(&mut painter, &mut ui, view, scheme, &mut actions)` draws everything and handles input
   as it goes; then `titlebar::draw(..)`.
4. `renderer.render(&instances, &mut atlas.uploads, &mut images.uploads, scale, palette.panel)`.
5. `ui.end()` clears one-shot input (`pressed`, `released`, wheel deltas).
6. Side effects collected as `Vec<Action>` are applied (`App::apply`); the cursor icon and IME area are
   pushed to the window; the next frame is requested if `ui.animating`.

**`Painter`** (`paint.rs:94-373`) is the only drawing API. It records `Instance`s and knows nothing
about input:

- Shapes: `rect(rect, color, radius)`, `bordered(rect, fill, radius, border, border_color)`,
  `gradient(rect, top, bottom, radius)`, `shadow(rect, color, radius, blur)`.
- Text: `layout(text, style, max_width) -> TextLayout`, `text(&layout, x, y, color)`,
  `text_aligned(..)`, `rich(..)`, `label(text, style, x, y, color) -> width`, `label_centered(..)`.
- Images: `image(path, rect, radius) -> Lookup`, `picture(..)`, `backdrop(..)`, `logo(..)`.
- Clipping: `push_clip(rect) -> previous` and `set_clip(previous)`; the clip rectangle is copied into
  every instance and enforced in the fragment shader. A cheap CPU cull drops instances fully outside
  the clip (`paint.rs:230-240`).
- Public fields: `fonts`, `scale`, `theme: Palette` (a copy; code reads `let t = p.theme;`),
  `atlas_full`.

**`Ui`** (`ui.rs:17-214`) is a plain struct of input and bookkeeping, passed as `&mut Ui` beside the
painter: `mouse`, `press_pos`, `down`, `pressed`, `released`, `clicks`, `scroll`, `scroll_x`, `mods`,
`dt`, `time`, `last_edit`, `focused`, `cursor`, `blocker`, `animating`, `still`, `held`.

**Widgets are free functions** of the shape `fn widget(p: &mut Painter, ui: &mut Ui, rect: Rect, ...)
-> bool` (clicked) that draw and hit-test in one call. The caller computes every rectangle by hand;
**there is no layout engine** (no flex, no grid, no auto-sizing beyond measuring a `TextLayout`).
Existing widgets: `ui::button` (styles `Primary`, `Secondary`, `Ghost`, `Danger`), `ui::icon_button`
(`ui.rs:384-439`); `widgets::{modal, line_field, field_label, dropdown, dropdown_frame, toggle, badge,
logo_badge, row, tile, chip}` and `widgets::Scroll` (`shell/widgets.rs`); `form::Fields` (a group of
bordered text fields sharing focus, `form.rs`); `color::ColorPicker` (`color.rs`). Larger surfaces are
methods on `Shell` in `shell/*.rs` that keep their own state structs.

**Ids.** `ui::id(key: impl Hash) -> u64` hashes any key with `DefaultHasher` (`ui.rs:253-259`). Ids
are used **only for animation state**, typically from a name plus position bits, for example
`id(("icon-button", rect.x.to_bits(), rect.y.to_bits()))`. They are not used for hit testing or focus.

**Hit testing.** `ui.hovered(rect)` is "mouse inside, and no scroll bar is being dragged
(`!ui.held`), and the mouse is not inside `ui.blocker`"; `ui.clicked(rect)` is "released this frame,
hovered, and the press started inside the rectangle" (`ui.rs:130-140`). There is no z-order
resolution: overlap is handled by `ui.blocker: Option<Rect>`, which the shell sets before drawing each
layer (the whole view under a modal; the open menu's rectangle from the previous frame; a floating
panel) and clears before drawing the overlay itself (`shell/mod.rs:1406-1449`). Overlays therefore
remember where they were drawn last frame (`MenuState::rect`, `ColorPicker::rect`).

**Focus.** No general focus system and no keyboard navigation between widgets. `Shell::key` routes a
key press by state, in this order: open menu, open modal, global shortcuts, title being renamed, find
bar, call, composer (`shell/mod.rs:1742-1951`). `form::Fields` keeps a `focus` index; Tab and
Shift+Tab cycle fields, Enter moves to the next single-line field (`form.rs:98-129`). The focused
field's border eases from `border_strong` to `border_focus`.

**Animation.** `ui.anim(id, target) -> f32` eases a stored value toward the target; `ui.appear(id)`
eases 0 to 1 from the first frame an id is drawn (`ui.rs:142-173`). Values not touched in a frame are
forgotten. Formula in section 16. Time-driven motion uses `ui.motion() -> Option<f32>` (`None` under
reduced motion).

**Scrolling** (`widgets::Scroll`, `shell/widgets.rs:17-199`). Protocol: `let offset =
scroll.update(ui, hovered)` before drawing the content, then `scroll.finish(ui, content_len,
view_len)` and `scroll.bar(p, ui, area, content_len)` after. The wheel moves a clamped `target`; `pos`
eases after it; the returned offset is rounded. Also `nudge`, `reveal`, `follow_end`, `jump_to_end`,
`settle`, `reset`, `at_end`, `progress`. Dragging the bar sets `ui.held`.

**Modals** (`widgets::modal`, `shell/widgets.rs:208-225`): dim the view with `scrim`, draw a centred
card of at most the requested size (clamped to `view.w - 48`, `view.h - 64`), return `(card,
clicked_outside)`. One modal at a time, stored as `Option<Modal>` on the shell.

**Menus** (`shell/menu.rs:722-883`): one open list at a time (`Option<MenuState>`), anchored to a
button rectangle, drawn after everything else. Rows are 34 px (48 px with a detail line), with an
optional icon, check mark or switch; an optional search field; at most 440 px tall with its own
`Scroll`; closes on a release outside, on Esc, or on a pick.

**Tooltips** (`ui.rs:175-251`): a widget calls `ui.tip(key, anchor, || Tip { text, rows })`; after the
mouse has rested 0.45 s the tip is stored and `ui::draw_tip` draws it last, below the anchor or above
it when there is no room. `control_flow` wakes the loop when a tooltip is due.

**Notices** (`shell/mod.rs:1717-1739`): up to three dismissible error cards stacked under the header.

**Threading with the UI.** UI code never blocks: it pushes `Action` values (`app.rs:192-348`);
`App::run` executes each on a named worker thread (`std::thread::Builder::new().name("atroxabot-worker")`)
whose result returns as a `WorkerEvent`. All file writes go through one writer thread (`Writer`,
`app.rs:1938-2002`).

**Snapshots** (`shell/snapshot.rs`, `gpu.rs:494-564`): the same UI code rendered offscreen to PNG, as
`#[ignore]`d tests run by hand:

- `ATROXABOT_SNAPSHOTS=<folder> cargo test -p atroxabot snapshots -- --ignored`; optional
  `ATROXABOT_SNAPSHOT_SCALE` (1 to 4).
- Each scene is a `Shell` filled with fake data. `frame(gpu, fonts, scheme, mouse, draw)`
  (`snapshot.rs:912-939`) runs the draw closure **five times with `dt = 1.0`** so every ease lands,
  decoding images synchronously in between, then renders the last frame's instances.
- `Offscreen::render` draws into an `Rgba8Unorm` texture (`RENDER_ATTACHMENT | COPY_SRC`), copies it
  to a `MAP_READ` buffer with rows padded to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`, maps it, and
  returns tightly packed RGBA bytes.
- Output: one PNG per scene and scheme, `{name}-{scheme}.png`, 1440 x 900 logical, written with the
  `png` crate (`write_png`, `snapshot.rs:1096-1103`).
- The app icon PNGs are produced the same way (`icon` test, `snapshot.rs:1111-1142`).
- A non-ignored test in the same file draws frames without a GPU and asserts that a busy session
  alone does not set `ui.animating` (`snapshot.rs:886-908`).

## 16. Design tokens

All from `crates/desktop/src/theme.rs` unless noted. Colours are sRGB hex; `@a` is alpha.
`Palette` is a `Copy` struct of 22 colours plus 6 syntax colours; `Palette::mix(&other, t)` blends two
palettes; `Palette::is_light()` is `bg.r + bg.g + bg.b > 1.5`. Schemes: `Scheme::{Black (default),
White, Amber}`, config keys `"black"`, `"white"`, `"amber"`.

### 16.1 Palettes (`theme.rs:102-184`)

| Field | Role | Black (default) | White | Amber |
|---|---|---|---|---|
| `bg` | main canvas background | `#121212` | `#FFFFFF` | `#F8F3EA` |
| `panel` | window, sidebar; also the clear colour | `#080808` | `#F4F4F5` | `#EFE8DC` |
| `surface` | raised: cards, menus, composer | `#1A1A1A` | `#FFFFFF` | `#FDF9F2` |
| `hover` | hovered element | `#1F1F1F` | `#ECECEE` | `#EAE1D2` |
| `active` | selected or pressed | `#282828` | `#E4E4E7` | `#E2D7C4` |
| `border` | hairline dividers | `#222222` | `#E6E6E9` | `#E5DBCA` |
| `border_strong` | controls, floating surfaces | `#303030` | `#D6D6DB` | `#D5C7B0` |
| `border_focus` | focused input | `#4C4C4C` | `#A1A1AA` | `#AE9A79` |
| `text` | primary text | `#EDEDED` | `#18181B` | `#2E2419` |
| `text_muted` | secondary text | `#A3A3A3` | `#55555E` | `#6E5F4C` |
| `text_faint` | hints, placeholders | `#6B6B6B` | `#9B9BA4` | `#A3927A` |
| `accent` | the single accent | `#EDEDED` | `#18181B` | `#BF6129` |
| `on_accent` | text on accent | `#0A0A0A` | `#FAFAFA` | `#FFF8EF` |
| `selection` | text selection | `#FFFFFF` @0.16 | `#2563EB` @0.16 | `#BF6129` @0.2 |
| `danger` | errors, destructive, removed lines | `#F26D6D` | `#DC2626` | `#B93A2C` |
| `positive` | added lines, success | `#6CCB8B` | `#15803D` | `#4D7C2A` |
| `merged` | merged state (purple) | `#B392F0` | `#8250DF` | `#7A4FA0` |
| `shadow` | drop shadows | `#000000` @0.6 | `#0F0F14` @0.09 | `#3C280F` @0.1 |
| `code_bg` | code block background | `#171717` | `#F6F6F7` | `#F1EADD` |
| `link` | links | `#7DB3FF` | `#2563EB` | `#A3501C` |
| `scrim` | modal dimmer, over pictures | `#000000` @0.6 | `#0C0C0F` @0.55 | `#1E140A` @0.55 |
| `on_scrim` | text on scrim | `#F5F5F5` | `#FFFFFF` | `#FFF9F0` |
| `syntax[0]` keyword | | `#FF7B72` | `#CF222E` | `#9D0006` |
| `syntax[1]` string | | `#A5D6FF` | `#0A3069` | `#79740E` |
| `syntax[2]` number | | `#79C0FF` | `#0550AE` | `#8F3F71` |
| `syntax[3]` comment | | `#8B949E` | `#6E7781` | `#928374` |
| `syntax[4]` function | | `#D2A8FF` | `#8250DF` | `#076678` |
| `syntax[5]` type | | `#FFA657` | `#953800` | `#B57614` |

In Black and White the accent is the text colour itself: the interface is greyscale. Rule
(`CLAUDE.md`): colours and sizes "live in `theme.rs`, from the active `Palette` (`p.theme`), never
hard-coded".

Colour-picker presets (`color.rs:19-22`): `#e5484d #f76b15 #ffc53d #46a758 #12a594 #00a2c7 #0090ff
#3e63dd #8e4ec6 #d6409f #a18072 #8b8d98`. Automatic identity colours:
`oklch(0.78, 0.115, hash % 360)` with FNV-1a over the name (`theme.rs:255-266`).

### 16.2 Radii and spacing (`theme.rs:342-359`)

| Constant | Value | Use |
|---|---|---|
| `RADIUS_SM` | 7.0 | list rows, buttons, text fields |
| `RADIUS` | 10.0 | menus, cards, tooltips, notices |
| `RADIUS_LG` | 16.0 | modals, the composer, large cards |
| `CANVAS_RADIUS` | 12.0 | the main canvas |
| `GAP` | 8.0 | window edge to canvas, between panels |
| `TITLE_HEIGHT` | 36.0 | title bar strip |
| `SIDEBAR_WIDTH` | 260.0 | sidebar (at most 40 % of the window) |
| `HEADER_HEIGHT` | 52.0 | header bar above the main area |
| `COLUMN_WIDTH` | 760.0 | maximum width of the reading column |

Other recurring measurements (module-local constants or literals): side panel width 320
(`shell/panel.rs:29`); field padding `FIELD_PAD = (12.0, 10.0)` (`form.rs:20`); title-bar button width
46, resize edge 5 (`titlebar.rs:17-19`); menu row 34 / 48, menu inner margin 6, menu max height 440
(`shell/menu.rs:741-757`); tooltip width 300, padding 12, row height 22 (`ui.rs:222-223`); modal margins
48 horizontal / 64 vertical (`shell/widgets.rs:214`); modal sizes 660 x 660, 940 x 660, 940 x 780;
sidebar rows 54 and 30 (`shell/sidebar.rs:19-21`); block gap 12, list indent 24, code padding 12
(`doc.rs:61-69`). Borders are 1.0 logical px (1.5 to 2.0 only for a few rings and markers). Selection
highlight radius 2.0. Caret: 2.0 wide (1.5 in single-line fields), accent colour, inset 3 px top and bottom.

Control sizes: switch 38 x 22, radius 11, knob 16 x 16 inset 3, travel 16 (`shell/widgets.rs:320-331`);
icon button icon 15 px; button height is the caller's (26, 28, 30 and 34 occur at call sites); chip
is a pill (`radius = h / 2`); badge radius `w * 0.28`, icon `w * 0.46`.

### 16.3 Type (`theme.rs:361-379`, `text.rs:34-52`)

| Style | Font | Size (logical px) | Line height |
|---|---|---|---|
| `BODY` | Inter Regular | 14.5 | 1.55 |
| `SMALL` | Inter Regular | 13.0 | 1.55 |
| `TINY` | Inter Regular | 11.5 | 1.55 |
| `LABEL` | Inter SemiBold | 13.0 | 1.3 |
| `CAPTION` | Inter SemiBold | 11.5 | 1.3 |
| `HEADING` | Inter SemiBold | 15.0 | 1.3 |
| `TITLE` | Inter SemiBold | 22.0 | 1.3 |
| `icon(size)` | Lucide | `size` | 1.0 |
| code (`doc.rs:71`) | JetBrains Mono | 13.0 (12.5 for details) | 1.6 |

Constructors: `Style::regular(size)` (1.55), `Style::semibold(size)` (1.3), `Style::mono(size)` (1.6).
Weights available: regular (400) and semibold (600) only. Typical icon sizes: 13, 14, 15, 16.

### 16.4 Shadows

`Painter::shadow(rect, color, radius, blur)` draws the rectangle blurred by `blur` logical px (alpha
`1 - smoothstep(-blur, blur, d)`). The convention is: draw the shadow first, offset downward (and for
large cards inset 8 px at each side), then the surface with a 1 px `border_strong` border.

| Surface | Offset y | Inset x | Colour | Radius | Blur | Evidence |
|---|---|---|---|---|---|---|
| Main canvas | +2 | 0 | `shadow` x 0.35 | 12 | 18 | `shell/mod.rs:1419-1420` |
| Tooltip | +6 | 0 | `shadow` x appear | 10 | 20 | `ui.rs:235` |
| Menu | +8 | 0 | `shadow` x appear | 10 | 28 | `shell/menu.rs:768` |
| Pick list above the composer | +8 | 0 | `shadow` | 10 | 24 | `shell/composer.rs:795` |
| Colour picker | +8 | 0 | `shadow` | 12 | 22 | `color.rs:154` |
| Notice card | +4 | 0 | `shadow` | 10 | 14 | `shell/mod.rs:1727` |
| Find bar | +5 | 0 | `shadow` x 0.8 | 10 | 18 | `shell/find.rs:233` |
| Composer | +10 | 8 | `shadow` x 0.8 | 16 | 28 | `shell/composer.rs:335-336` |
| Approval / question card | +8 | 8 | `shadow` x 0.7 | 16 | 22 | `shell/composer.rs:875-876` |
| Small tray | +6 | 6 | `shadow` x 0.45 | 10 | 16 | `shell/queue.rs:158-163` |
| Modal | +16 | 0 | `shadow` x appear | 16 | 48 | `shell/widgets.rs:221` |

Glows are the same primitive in a colour (`orb.rs`).

### 16.5 Motion

| What | Value | Evidence |
|---|---|---|
| Standard ease (hover, focus, toggle, appear) | exponential approach, per frame `value += (target - value) * (1 - exp(-dt * 16))`; snaps when within 0.002 | `ui.rs:162-173` |
| Scroll ease | `pos += (target - pos) * (1 - exp(-dt * 22))`; snaps when within 0.5 px; drawn offset rounded | `shell/widgets.rs:15`, `74-81` |
| Theme cross-fade | 0.25 s, smoothstep `p * p * (3 - 2p)` over `Palette::mix` | `app.rs:48`, `1496-1509` |
| Frame `dt` clamp | 0.05 s | `app.rs:1492` |
| Modal entrance | fades with `appear`, slides up from +10 px | `shell/widgets.rs:212-218` |
| Tooltip delay | 0.45 s | `ui.rs:70` |
| Caret blink | period 1.0 s, visible for the first 0.55 s, restarts on every edit | `ui.rs:204-213` |
| Double / triple click | within 0.4 s and 4 px | `ui.rs:110-111` |
| Title-bar double press | 0.4 s | `app.rs:713` |
| Hover fills | `fade(t.hover, hover)`; text `mix(t.text_muted, t.text, hover)`; primary button `fade(t.accent, 1.0 - 0.12 * hover)` | `ui.rs:386-439` |
| Spinner | 4 px dot on a 13 px ring, 5 rad/s; the ring at 25 % alpha | `shell/thread.rs:169-174` |
| Pulsing dot | alpha `0.35 + 0.65 * (0.5 + 0.5 * sin(4t))` | `shell/thread.rs:143-145` |
| Background "busy" redraw | about 60 fps on whole refreshes; 1 fps under reduced motion | `app.rs:63`, `1597-1610` |

There are no duration-based tweens, no easing-curve library, no springs.

### 16.6 Scroll bars (`shell/widgets.rs:34-206`)

- Overlay style: no gutter, no track drawn, no arrows. The thumb is drawn only when content overflows.
- Hit zone: a 12 px strip (`ZONE`) along the right edge (or the bottom for horizontal).
- Track: inset 8 px from both ends (`len = view - 16`).
- Thumb length `max(track * view / content, 28)`, capped at the track.
- Thumb thickness 3 px, growing to 6 px while hovered or dragged (eased), 4 px from the edge, growing
  inward; radius `thickness / 2`.
- Thumb colour `text_faint` at alpha 0.35, rising to 0.75.
- A press on the thumb drags it; a press beside it first centres the thumb there. While dragged the
  view follows at once (no easing) and `ui.held` suppresses every other hover and click.
- Wheel: one line is 100 logical px; the target is clamped to the content, so nothing overscrolls.

## 17. wgpu 30 / winit 0.30 as actually used (cheat-sheet)

All excerpts are verbatim from the reference (trimmed with `// ...`). wgpu 30.0.1, winit 0.30.13.

**Instance, surface, adapter, device** (`crates/desktop/src/gpu.rs:141-162`):

```rust
pub async fn new(window: Arc<Window>, display: winit::event_loop::OwnedDisplayHandle) -> Result<Self, GpuError> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)));
    let size = window.inner_size();
    let surface = instance.create_surface(window).map_err(GpuError::Surface)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(GpuError::Adapter)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("atroxabot"),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        })
        .await
        .map_err(GpuError::Device)?;
```

Notes: `Instance::new` takes the descriptor **by value**. `InstanceDescriptor` has no `Default`; the
constructors are `new_with_display_handle(Box<dyn ..>)`, `new_without_display_handle()` and their
`_from_env` variants. `request_adapter` returns `Result<Adapter, wgpu::RequestAdapterError>` (not
`Option`). `request_device` takes one argument. Error types: `wgpu::CreateSurfaceError`,
`wgpu::RequestAdapterError`, `wgpu::RequestDeviceError`. Other `DeviceDescriptor` fields (left
default): `required_features`, `experimental_features`, `trace`. Other `RequestAdapterOptions` fields:
`force_fallback_adapter`, `apply_limit_buckets`.

**Error and device-lost callbacks** (`gpu.rs:163-175`):

```rust
let logged = AtomicU32::new(0);
device.on_uncaptured_error(Arc::new(move |error| {
    if logged.fetch_add(1, Ordering::Relaxed) < MAX_LOGGED_ERRORS {
        crate::log::error(format!("GPU error: {error}"));
    }
}));
let lost = Arc::new(AtomicBool::new(false));
let flag = Arc::clone(&lost);
device.set_device_lost_callback(move |reason, message| {
    crate::log::error(format!("GPU device lost ({reason:?}): {message}"));
    flag.store(true, Ordering::Relaxed);
});
```

Error scopes (`push_error_scope`) are not used by the reference.

**Surface configuration** (`gpu.rs:179-198`):

```rust
let caps = surface.get_capabilities(&adapter);
let format = caps
    .formats
    .iter()
    .copied()
    .find(|f| !f.is_srgb())
    .or_else(|| caps.formats.first().copied())
    .ok_or(GpuError::NoSurfaceFormat)?;
let config = wgpu::SurfaceConfiguration {
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    format,
    color_space: wgpu::SurfaceColorSpace::Auto,
    width: size.width.max(1),
    height: size.height.max(1),
    present_mode: wgpu::PresentMode::AutoVsync,
    desired_maximum_frame_latency: 2,
    alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
    view_formats: Vec::new(),
};
surface.configure(&device, &config);
```

Note the `color_space` field, new in recent wgpu.

**Acquiring and presenting a frame** (`gpu.rs:243-267`):

```rust
let frame = match self.surface.get_current_texture() {
    wgpu::CurrentSurfaceTexture::Success(frame) => frame,
    wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
        // Draw this one, reconfigure for the next.
        self.surface.configure(&self.device, &self.config);
        frame
    }
    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
        self.surface.configure(&self.device, &self.config);
        // Once at once; a second time in a row it waits like a skipped frame.
        return if std::mem::replace(&mut self.retried, true) { Frame::Skipped } else { Frame::Retry };
    }
    wgpu::CurrentSurfaceTexture::Timeout
    | wgpu::CurrentSurfaceTexture::Occluded
    | wgpu::CurrentSurfaceTexture::Validation => {
        return Frame::Skipped;
    }
};
let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
// ... encode ...
self.queue.submit([commands]);
self.queue.present(frame);
```

`get_current_texture()` returns the **enum** `wgpu::CurrentSurfaceTexture` (seven variants), not a
`Result<SurfaceTexture, SurfaceError>`. Presenting is `queue.present(frame)`.

**Shader module, buffers, textures, sampler** (`gpu.rs:274-309`):

```rust
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("ui"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});
let globals = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("globals"),
    size: size_of::<Globals>() as u64,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
device.create_texture(&wgpu::TextureDescriptor {
    label: Some(label),
    size: wgpu::Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1 },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format,
    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    view_formats: &[],
})
let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
    label: Some("atlas"),
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    ..Default::default()
});
```

**Bind group layout and bind group** (`gpu.rs:311-361`, entries abbreviated):

```rust
let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("ui"),
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // ... binding 3: a second texture, as binding 1 ...
    ],
});
let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("ui"),
    layout: &layout,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas_view) },
        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&images_view) },
    ],
});
```

**Pipeline layout and render pipeline** (`gpu.rs:363-397`):

```rust
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("ui"),
    bind_group_layouts: &[Some(&layout)],
    immediate_size: 0,
});
let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4];
let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("ui"),
    layout: Some(&pipeline_layout),
    vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        buffers: &[Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &attributes,
        })],
    },
    primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
    depth_stencil: None,
    multisample: wgpu::MultisampleState::default(),
    fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        targets: &[Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })],
    }),
    multiview_mask: None,
    cache: None,
});
```

Version-sensitive names: `bind_group_layouts: &[Some(&layout)]` (slice of `Option`),
`immediate_size: 0` (was `push_constant_ranges`), `buffers: &[Some(VertexBufferLayout { .. })]`
(slice of `Option`), `entry_point: Some(..)`, `multiview_mask: None` (was `multiview`), `cache: None`.

**Writes** (`gpu.rs:416-430`, `453-461`):

```rust
queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
let bytes = bytemuck::cast_slice::<Instance, u8>(instances);
queue.write_buffer(&self.instances, 0, bytes);

queue.write_texture(
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: upload.x, y: upload.y, z: 0 },
        aspect: wgpu::TextureAspect::All,
    },
    &upload.data,
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(upload.w * bytes_per_texel),
        rows_per_image: None,
    },
    wgpu::Extent3d { width: upload.w, height: upload.h, depth_or_array_layers: 1 },
);
```

**Encoder and render pass** (`gpu.rs:463-490`):

```rust
let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui") });
{
    let [r, g, b, a] = clear.map(f64::from);
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    if !instances.is_empty() {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..bytes.len() as u64));
        pass.draw(0..4, 0..instances.len() as u32);
    }
}
encoder.finish()
```

`RenderPassColorAttachment` has a `depth_slice` field. The remaining `RenderPassDescriptor` fields
(left default) are `depth_stencil_attachment`, `timestamp_writes`, `occlusion_query_set`,
`multiview_mask`.

**Readback** (test-only, `gpu.rs:541-562`):

```rust
let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("snapshot"),
    size: u64::from(row * height),
    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    mapped_at_creation: false,
});
let mut copy = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snapshot") });
copy.copy_texture_to_buffer(
    texture.as_image_copy(),
    wgpu::TexelCopyBufferInfo {
        buffer: &buffer,
        layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
    },
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
);
self.queue.submit([draw, copy.finish()]);
buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
let mapped = buffer.slice(..).get_mapped_range().expect("the snapshot buffer was mapped");
mapped.chunks(row as usize).flat_map(|line| &line[..(width * 4) as usize]).copied().collect()
```

`device.poll` takes `wgpu::PollType` and returns a `Result`; `get_mapped_range()` returns a `Result`
in wgpu 30.

**Shader validation in a test** (`gpu.rs:572-581`): `use wgpu::naga;` then
`naga::front::wgsl::parse_str(src)` and
`naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module)`.

**Block-on helper** (`crates/desktop/src/app.rs:2036-2053`), used as
`block_on(Renderer::new(Arc::clone(&window), display.clone()))`:

```rust
/// Minimal executor for wgpu's initialisation futures.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Arc::new(Unpark(std::thread::current())).into();
    let mut cx = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}
```

**winit `ApplicationHandler`** (`crates/desktop/src/main.rs:48-106`, `158-163`):

```rust
struct Handler {
    app: Option<App>,
    proxy: EventLoopProxy<WorkerEvent>,
    error: Option<app::StartupError>,
}

impl ApplicationHandler<WorkerEvent> for Handler {
    fn new_events(&mut self, _: &ActiveEventLoop, cause: StartCause) {
        if let (StartCause::ResumeTimeReached { .. }, Some(app)) = (cause, &mut self.app) {
            app.wake();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        match App::new(event_loop, self.proxy.clone()) {
            Ok(app) => self.app = Some(app),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: WorkerEvent) {
        if let Some(app) = &mut self.app {
            app.worker_event(event_loop, event);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(app) = &mut self.app {
            app.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(app) = &self.app {
            event_loop.set_control_flow(app.control_flow());
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // ... take what must outlive the app, then:
        self.app = None;
    }
}

// in main():
let event_loop = match EventLoop::<WorkerEvent>::with_user_event().build() {
    Ok(event_loop) => event_loop,
    Err(e) => return fail(&format!("The window system could not be started: {e}")),
};
let mut handler = Handler { app: None, proxy: event_loop.create_proxy(), error: None };
let result = event_loop.run_app(&mut handler);
```

**Window creation inside `resumed`** (`app.rs:474-506`):

```rust
let attributes = Window::default_attributes()
    .with_title("AtroxaBot")
    .with_inner_size(LogicalSize::new(width, height))
    .with_min_inner_size(min_size(zoom, event_loop.primary_monitor()))
    .with_maximized(maximized && !cfg!(windows))
    .with_theme(Some(window_theme(scheme)))
    .with_visible(false)
    .with_decorations(!crate::titlebar::CUSTOM);
#[cfg(windows)]
let attributes =
    winit::platform::windows::WindowAttributesExtWindows::with_undecorated_shadow(attributes, true);
// ... macOS and Linux attribute extensions ...
let window = Arc::new(event_loop.create_window(attributes).map_err(StartupError::Window)?);
window.set_ime_allowed(true);
let display = event_loop.owned_display_handle();
let renderer = block_on(Renderer::new(Arc::clone(&window), display.clone())).map_err(StartupError::Gpu)?;
```

**Other winit calls used:** `window.request_redraw()`, `window.inner_size()`,
`window.scale_factor()`, `window.set_cursor(CursorIcon)`, `window.set_visible(bool)`,
`window.set_minimized(bool)`, `window.is_minimized() -> Option<bool>`, `window.set_maximized(bool)`,
`window.is_maximized()`, `window.fullscreen()`, `window.drag_window()`,
`window.drag_resize_window(ResizeDirection)`, `window.set_min_inner_size(Some(..))`,
`window.set_theme(Some(Theme))`, `window.set_ime_cursor_area(LogicalPosition, LogicalSize)`,
`window.focus_window()`, `window.request_user_attention(..)`, `window.current_monitor()`,
`MonitorHandle::refresh_rate_millihertz()`, `event_loop.exit()`,
`event_loop.set_control_flow(ControlFlow::Wait | ControlFlow::WaitUntil(Instant))`,
`EventLoopProxy::send_event(..)`. Window events matched: `CloseRequested`, `RedrawRequested`,
`Resized`, `Occluded`, `Moved`, `Focused`, `DroppedFile`, `ModifiersChanged`, `CursorMoved`,
`CursorLeft`, `MouseInput`, `MouseWheel`, `KeyboardInput`, `Ime`.

## 18. Reusable code and licence

**Licence.** `license = "MIT OR Apache-2.0"` (`Cargo.toml:9`); `LICENSE-MIT` reads "Copyright (c) 2026
Luvarly and the OpenRP contributors"; `LICENSE-APACHE` is present. The README adds: "The renderer,
text engine and widgets come from OpenRP (https://github.com/Luvarly/openrp), under the same
licenses." Either licence permits copying with the notice kept.

**No shared crate exists.** The renderer, text engine and toolkit live inside the binary crate
`atroxabot` (`crates/desktop`, `[[bin]]` only), so they cannot be a dependency. Under REF-03 reuse
means **copying with attribution** (record source path, commit `a1c5dc5` and the licence line).

Bundled assets keep their own licences (texts in `crates/desktop/assets/*-LICENSE.txt`): Inter,
Literata and JetBrains Mono under the SIL Open Font License; Twemoji Mozilla (Apache 2.0 code, CC-BY
4.0 graphics); Lucide (ISC). The brand outlines in `assets/logos/` are Simple Icons (CC0) but depict
third-party trademarks and are not for reuse.

Candidates, most self-contained first:

| File or function | Lines | What it gives | Coupling to remove |
|---|---|---|---|
| `app.rs` `block_on` | 18 | executor for wgpu init futures | none |
| `atlas.rs` | 140 | shelf packer with LRU eviction, tested | none |
| `raster.rs` | 266 | analytic anti-aliased outline rasteriser, tested | `ttf-parser` types |
| `editor.rs` | 314 | text-editing state (caret, selection, clusters), tested | `crate::font::is_emoji_joiner` |
| `font.rs` | 490 | faces, kerning, ligatures, COLR layers, system fallback | embedded font list |
| `text.rs` | 841 | layout, wrapping, hit testing, glyph atlas | `font`, `atlas`, `raster` |
| `gpu.rs` + `shader.wgsl` | 582 + 139 | the 2D quad renderer, `Frame` handling, device-lost flag, `Offscreen` readback | `crate::log`, `text::{ATLAS_SIZE, Upload}` |
| `paint.rs` | 373 | `Painter`, `Rect`, `Color` helpers | `image`, `logo` (droppable) |
| `ui.rs` | 603 | `Ui`, `id`, `anim`, tooltips, `button`, `icon_button`, `edit_key`, icon constants | `theme`, `editor`, `arboard` |
| `theme.rs` | 422 | `Palette`, schemes, sizes, text styles | `agent_color`, `provider_color` are app-specific |
| `shell/widgets.rs` | 491 | `Scroll`, `modal`, `line_field`, `dropdown`, `toggle`, `chip`, `row`, `tile` | items are `pub(super)`; `logo_badge` is app-specific |
| `form.rs` | 236 | grouped text fields with focus, selection, IME caret | `editor`, `ui`, `theme` |
| `color.rs` | 341 | colour picker popover; HSV conversions | `theme::agent_color` call |
| `titlebar.rs` | 145 | custom title bar: grab zones, window buttons, resize cursors | `theme::TITLE_HEIGHT` |
| `log.rs` | 108 | file logger thread and crash-report panic hook | `atroxabot_core::{Config::dir, unix_now}`, file names |
| `platform.rs` | 744 | `open_url`, `open_folder`, `alert`, `reduce_motion`, `pick_files`, `pick_folder`, `clipboard_files`, `run_in_terminal`, `no_window` | `ATROXABOT_*` environment variable names; needs `arboard` and `png` |
| `build.rs` | 128 | Windows icon and version resource written as a `.res` by hand (MSVC only) | product strings |
| `app.rs` window code | | `window_state`, `min_size`, `zoomed`, `refresh_rate`, `busy_interval`, `set_icon`, `window_theme`, `control_flow` pattern, `Writer` thread | embedded in `App` |
| `image.rs` | 295 | image atlas with worker decode | `atroxabot_core::picture` |
| `shell/snapshot.rs` `frame`, `write_png` | ~40 | offscreen scene harness | `Shell` |
| `packaging/` | | Inno Setup script, `Info.plist`, `.desktop` + `install.sh`, release and signing scripts | product names and ids |

Gaps a copier must fill: the file pickers are **open-only** (no save dialog); there is no right-click,
no touch, no keyboard focus traversal, no layout engine, no multi-window.

## 19. Other conventions (naming, module style, tests) and gaps

**Module and naming style** (relevant to REF-02):

- Flat modules: one file per concern directly in `src/`, declared alphabetically with `mod x;` in
  `main.rs` (`main.rs:11-33`); the one sub-tree uses `shell/mod.rs`. Short, single-word module names
  (`gpu`, `paint`, `text`, `font`, `atlas`, `raster`, `theme`, `ui`, `form`, `editor`, `platform`,
  `log`).
- Every file starts with a `//!` module comment explaining what it is and why; every item, including
  private fields worth explaining, has a `///` comment. `# Errors` and `# Panics` sections are used.
- Types are plain nouns (`Renderer`, `Painter`, `Ui`, `Rect`, `Fonts`, `GlyphAtlas`, `TextLayout`,
  `Palette`, `Scheme`, `Editor`, `Scroll`); functions are verbs or nouns in `snake_case`; constants
  `SCREAMING_SNAKE_CASE` with a doc comment and units.
- Conventional local names: `p` (painter), `ui`, `t` (the palette, `let t = p.theme;`), `s` (scale).
- British spelling in prose and identifiers of the text domain ("colour", "rasterise", "centre").
- Known shortcuts are marked with a `// ponytail:` comment stating the limitation and the upgrade path
  (`font.rs:14`, `editor.rs:4`, `platform.rs:8`).
- Lint exceptions always carry a reason:
  `#[allow(clippy::too_many_arguments, reason = "...")]`, `#[expect(..., reason = "...")]`.
- `unsafe` is confined to hand-written FFI, each block with a `// SAFETY:` comment; FFI is declared
  locally as `#[link(name = "user32")] unsafe extern "system" { .. }`.
- Rust 2024 idioms are used freely: let-chains (`if let .. && let ..`), `f32::midpoint`,
  `is_none_or`, `is_some_and`, `std::pin::pin!`.
- Threads are always named (`"atroxabot-worker"`, `"atroxabot-writer"`, `"atroxabot-log"`); spawn
  failures are logged, not unwrapped. No async runtime.
- State on disk lives under `~/.atroxabot/`; writes are atomic and `0600` on Unix.

**Tests:** inline `#[cfg(test)] mod tests` at the bottom of each file (184 `#[test]`s in the desktop
crate); names are sentences (`a_tooltip_shows_once_the_mouse_has_rested`,
`packs_rows_and_evicts_the_least_recently_used`); assertions carry messages. Tests needing a GPU, a
window, a console or real services are `#[ignore = "reason"]` and run by hand. Rule (`CLAUDE.md`):
"non-trivial logic gets a small unit test. Parsers survive hostile input."

**Hand-written network client style** (`crates/mcp/src/`, an MCP client over stdio and Streamable
HTTP; the model for any protocol code):

- Blocking I/O on plain named threads, no async. HTTP through `ureq` 3 with an explicit agent:
  `ureq::Agent::config_builder().http_status_as_error(false).max_redirects(0)
  .timeout_connect(Some(Duration::from_secs(15))).user_agent(concat!("atroxabot/",
  env!("CARGO_PKG_VERSION"))).build().into()` (`http.rs:60-67`). Status codes and redirects are
  handled by hand (only 307/308 within the same origin, at most 3 hops, `http.rs:180-198`, `249-261`).
- Every wait is bounded and cancellable: a request runs on its own thread (`"mcp-http"`) and reports
  through `mpsc::sync_channel(1)`; the caller waits with `conn::wait(&rx, deadline, &stop)`, which
  polls `recv_timeout` every 25 ms against a deadline and two `AtomicBool` stop flags
  (`conn.rs:27-60`, `http.rs:74-104`).
- JSON-RPC is built and read as untyped `serde_json::Value` with `json!` (`proto::request`,
  `proto::notification`, `proto::classify`); no serde derive structs in that crate. Unknown fields and
  shapes are tolerated; ids are numeric `u64`.
- Everything read from outside is capped by named constants (`MAX_LINE = 8 << 20`, `MAX_TOOLS = 512`,
  `MAX_PAGES = 32`, `MAX_TEXT = 200_000`, `proto.rs:20-30`), and bodies are read with
  `.take(limit).read_to_end(..)`.
- Server-sent events and line framing are parsed by hand (`proto::Sse`, `proto::read_line`).
- Secrets never reach `Display` or `Debug`: `Transport` has a manual `Debug` that prints header and
  environment names without values and drops the URL query (`lib.rs:74-92`); error messages never echo
  an address (`http.rs:263-270`). Addresses are validated up front (`check_url`: https anywhere, plain
  http only for localhost, no credentials in the URL).
- The crate's error enum is `pub(crate)`; the public API returns strings or simple data
  (`Hub::tools() -> (Vec<Tool>, Vec<(String, String)>)`). Mapping from library errors is done by small
  functions (`network_error`, `read_error`).
- Tests are tables of accepted and refused inputs (`http.rs:281-334`); end-to-end tests live in
  `crates/mcp/tests/`.

**Gaps (things the specification must decide because the reference has no equivalent):**

| Topic | Status in the reference |
|---|---|
| 3D rendering, depth buffer, MSAA, multiple passes, render targets | none (one 2D pass to the surface) |
| Render graph | none |
| Compute shaders, storage buffers, storage textures | none; the requested limits (`downlevel_webgl2_defaults`) forbid them, so HumanArt must request different limits |
| wgpu features | none requested |
| Mesh vertex buffers, index buffers | none (instanced quads from `vertex_index`) |
| Mipmaps, texture arrays, cube maps, compressed or HDR formats | none |
| Linear-light / HDR colour pipeline, tone mapping | none (sRGB-space blending by design) |
| Math types | none |
| Shader organisation beyond one file; includes; hot reload | none |
| Pipeline cache | none (`cache: None`) |
| Web / wasm build | none |
| Multi-window, touch, right-click, gamepad | none |
| Log levels, structured logging | none (single `error` level) |
| Save-file dialog | none |
| GPU timing / profiling | none |
