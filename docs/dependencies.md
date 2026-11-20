# Dependencies

Every third-party crate, why it is there, and why an in-house implementation was not reasonable
(`[DEP-03]`). The default answer to a new crate is no. `tools/check-deps.sh` fails when `cargo tree`
shows a crate that the list at the end of this file does not record (`[DEP-06]`).

## Direct dependencies

| Crate | Version | Licence | What it does | Why not in-house |
|---|---|---|---|---|
| `wgpu` | 30.0.x | MIT OR Apache-2.0 | The GPU API: Vulkan, Metal, DX12, GL and WebGPU behind one interface, and the WGSL compiler (naga). | Named by the specification (`[DEP-01]`). Three native graphics APIs and a shader compiler are several person-years of work. |
| `winit` | 0.30.x | Apache-2.0 | Windows, the event loop, keyboard, mouse, touch and IME on every desktop system. | The reference application's windowing crate (`[DEP-01]`, Appendix F). Five windowing systems are not reasonable to rewrite. |

Nothing else is a direct dependency of any crate in the workspace, in any dependency table (normal,
dev or build). `tools/check-tree.sh` enforces that.

## Transitive dependencies

`wgpu` and `winit` bring their own dependencies (`[DEP-06]` permits them). Counted over every target
the workspace builds for (Windows, macOS, Linux, wasm32) there are 220 crates in total, the two
above included; a Windows build of the editor links fewer than half of them. Some of the transitive
crates have names that `[DEP-02]` forbids as direct dependencies (`log`, `thiserror`, `bytemuck`,
`tokio` and `tracing` come with wgpu's and winit's platform back ends): no HumanArt crate may name
them or use them, and none does.

## Recorded crates

```
ab_glyph
ab_glyph_rasterizer
ahash
allocator-api2
android-activity
android-properties
android_system_properties
arrayref
arrayvec
as-raw-xcb-connection
ash
atomic-waker
autocfg
bit-set
bit-vec
bitflags
block2
bumpalo
bytemuck
bytemuck_derive
bytes
calloop
calloop-wayland-source
cc
cfg-if
cfg_aliases
codespan-reporting
combine
concurrent-queue
core-foundation
core-foundation-sys
core-graphics
core-graphics-types
crossbeam-utils
crunchy
cursor-icon
dispatch
dlib
document-features
downcast-rs
dpi
equivalent
errno
find-msvc-tools
foldhash
foreign-types
foreign-types-macros
foreign-types-shared
futures-core
futures-task
futures-util
gethostname
getrandom
gl_generator
glow
glutin_wgl_sys
gpu-allocator
half
hashbrown
hermit-abi
indexmap
jni
jni-macros
jni-sys
jni-sys-macros
jobserver
js-sys
khronos-egl
khronos_api
libc
libloading
libm
libredox
linux-raw-sys
litrs
lock_api
log
memchr
memmap2
naga
naga-types
ndk
ndk-context
ndk-sys
num-traits
num_enum
num_enum_derive
objc-sys
objc2
objc2-app-kit
objc2-core-foundation
objc2-core-graphics
objc2-encode
objc2-foundation
objc2-metal
objc2-quartz-core
objc2-ui-kit
once_cell
orbclient
ordered-float
owned_ttf_parser
parking_lot
parking_lot_core
percent-encoding
pin-project
pin-project-internal
pin-project-lite
pkg-config
plain
polling
portable-atomic
portable-atomic-util
presser
proc-macro-crate
proc-macro2
profiling
quick-xml
quote
r-efi
range-alloc
raw-window-handle
raw-window-metal
redox_syscall
renderdoc-sys
rustc-hash
rustc_version
rustix
rustversion
same-file
scoped-tls
scopeguard
sctk-adwaita
semver
shlex
simd_cesu8
simdutf8
slab
slotmap
smallvec
smithay-client-toolkit
smol_str
spirv
static_assertions
strict-num
syn
thiserror
thiserror-impl
tiny-skia
tiny-skia-path
tokio
toml_datetime
toml_edit
toml_parser
tracing
tracing-core
ttf-parser
unicode-ident
unicode-segmentation
unicode-width
version_check
walkdir
wasip2
wasm-bindgen
wasm-bindgen-futures
wasm-bindgen-macro
wasm-bindgen-macro-support
wasm-bindgen-shared
wayland-backend
wayland-client
wayland-csd-frame
wayland-cursor
wayland-protocols
wayland-protocols-plasma
wayland-protocols-wlr
wayland-scanner
wayland-sys
web-sys
web-time
wgpu
wgpu-core
wgpu-core-deps-apple
wgpu-core-deps-emscripten
wgpu-core-deps-windows-linux-android
wgpu-hal
wgpu-naga-bridge
wgpu-types
winapi-util
windows
windows-collections
windows-core
windows-future
windows-implement
windows-interface
windows-link
windows-numerics
windows-result
windows-strings
windows-sys
windows-targets
windows-threading
windows_aarch64_gnullvm
windows_aarch64_msvc
windows_i686_gnu
windows_i686_gnullvm
windows_i686_msvc
windows_x86_64_gnu
windows_x86_64_gnullvm
windows_x86_64_msvc
winit
winnow
wit-bindgen
x11-dl
x11rb
x11rb-protocol
xcursor
xkbcommon-dl
xkeysym
xml-rs
zerocopy
zerocopy-derive
```
