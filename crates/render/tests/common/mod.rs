//! Shared helpers of the GPU tests: one headless device, a renderer on a scene, frame loops,
//! readbacks and PNG output under target/agents/render/shots/.
//!
//! One device per test binary, created once and never dropped: creating and dropping devices on
//! several test threads at once deadlocked the NVIDIA driver under GPU contention (a thread exit
//! holding the loader lock while `vkCreateDevice` / WGL waited for it, see the skin-shading
//! STATUS). A watchdog started before the device aborts the binary with a message when no test
//! makes progress for [`WATCHDOG_LIMIT`], so a hang becomes a failure instead of an idle process.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use humanart_gpu::resources::{Budget, Resources, TextureId};
use humanart_gpu::{DeviceRequest, Gpu};
use humanart_render::camera::Camera;
use humanart_render::capture;
use humanart_render::light::LightRig;
use humanart_render::renderer::{FrameInput, OverlayPrimitive, RenderConfig, Renderer, StillFrame};
use humanart_render::scene::{GpuScene, Scene};

/// How long a blocking wait for the GPU may take before the test fails.
pub const GPU_WAIT_LIMIT: Duration = Duration::from_secs(120);

/// How long the binary may go without progress (a frame, a readback, a new harness) before the
/// watchdog aborts it.
pub const WATCHDOG_LIMIT: Duration = Duration::from_secs(300);

/// Milliseconds since the watchdog started at the last progress mark.
static PROGRESS_MS: AtomicU64 = AtomicU64::new(0);

/// The start of the watchdog's clock.
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Marks progress for the watchdog.
pub fn progress() {
    if let Some(epoch) = EPOCH.get() {
        PROGRESS_MS.store(epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
}

/// Starts the watchdog thread once. It must run before any GPU work: in a loader-lock deadlock no
/// new thread can start. It aborts (no `exit`: that would take the loader lock again).
fn start_watchdog() {
    let mut started = false;
    let epoch = *EPOCH.get_or_init(|| {
        started = true;
        Instant::now()
    });
    if !started {
        return;
    }
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let idle = epoch.elapsed().saturating_sub(Duration::from_millis(PROGRESS_MS.load(Ordering::Relaxed)));
            if idle > WATCHDOG_LIMIT {
                eprintln!("watchdog: no test progress for {} s, aborting the test binary", idle.as_secs());
                std::process::abort();
            }
        }
    });
}

/// The test binary's one headless device (created on first use, never dropped).
pub fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    start_watchdog();
    let gpu = GPU.get_or_init(|| Gpu::headless_blocking(&DeviceRequest::default()).expect("this machine has a GPU"));
    progress();
    gpu
}

/// Waits for every submission so far, at most [`GPU_WAIT_LIMIT`].
pub fn wait(device: &wgpu::Device) {
    let poll = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(GPU_WAIT_LIMIT) });
    if let Err(e) = poll {
        panic!("the GPU did not finish within {} s: {e}", GPU_WAIT_LIMIT.as_secs());
    }
    progress();
}

pub fn shots_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/render/shots");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn save_png(name: &str, width: u32, height: u32, rgba: &[u8]) {
    let png = capture::encode_png_rgba8(width, height, rgba).unwrap();
    std::fs::write(shots_dir().join(name), png).unwrap();
}

pub struct Harness {
    pub gpu: &'static Gpu,
    pub resources: Resources,
    pub renderer: Renderer,
    pub scene: GpuScene,
}

impl Harness {
    pub fn new(config: RenderConfig, scene: &Scene) -> Self {
        let gpu = gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let renderer = Renderer::new(gpu, &mut resources, config).unwrap();
        let scene = GpuScene::new(&mut resources, scene).unwrap();
        Self { gpu, resources, renderer, scene }
    }

    /// Renders one frame and submits it.
    pub fn frame(
        &mut self,
        camera: &Camera,
        rig: &LightRig,
        overlay: &[OverlayPrimitive],
        wireframe: bool,
        still: Option<StillFrame>,
        reset: bool,
    ) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test frame") });
        self.renderer
            .render(
                &mut self.resources,
                &mut encoder,
                FrameInput {
                    camera,
                    rig,
                    scene: &mut self.scene,
                    overlay,
                    wireframe,
                    still,
                    pick: None,
                    reset_history: reset,
                },
            )
            .unwrap();
        self.gpu.queue.submit([encoder.finish()]);
        self.renderer.after_submit();
        progress();
    }

    /// Renders `n` frames of a still camera (TAA converging).
    pub fn frames(&mut self, camera: &Camera, rig: &LightRig, n: u32) {
        for i in 0..n {
            self.frame(camera, rig, &[], false, None, i == 0);
        }
    }

    pub fn read(&mut self, texture: TextureId) -> (u32, u32, Vec<u8>) {
        let read = capture::read_texture(&mut self.resources, texture, 0).unwrap();
        progress();
        read
    }

    pub fn display(&mut self) -> (u32, u32, Vec<u8>) {
        let t = self.renderer.display_texture().unwrap();
        self.read(t)
    }

    pub fn resolved(&mut self) -> Vec<[f32; 4]> {
        let t = self.renderer.resolved_texture().unwrap();
        let (_, _, bytes) = self.read(t);
        capture::rgba16f_to_f32(&bytes)
    }

    pub fn save_display(&mut self, name: &str) -> (u32, u32, Vec<u8>) {
        let (w, h, rgba) = self.display();
        save_png(name, w, h, &rgba);
        (w, h, rgba)
    }
}
