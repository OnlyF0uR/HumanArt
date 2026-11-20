//! Instance, adapter and device creation, the limits request `[GPU-02]`, the capability model
//! `[PLAT-05]` (spec 14.5) and surface configuration.
//!
//! Device setup follows the reference application (`docs/reference-conventions.md` sections 1, 2):
//! the instance takes its backends from the compiled-in features and the `WGPU_BACKEND` environment
//! variable, uncaptured errors and device loss go to callbacks instead of panicking, the surface
//! format is the first non-sRGB one, the present mode is `AutoVsync`, and resizes are deferred to the
//! next frame. Three things differ, because the reference is a UI and this is a renderer, and the
//! reference has no equivalent for them:
//!
//! - the limits are the ones of `[GPU-02]` clamped to the adapter, not the WebGL2 floor;
//! - optional features are requested through the capability model;
//! - the default power preference is `HighPerformance` (the reference asks for `LowPower` because "a
//!   UI does not need the discrete card"). `WGPU_POWER_PREF` overrides it, and
//!   [`DeviceRequest::reference_ui`] gives the reference's exact values.
//!
//! This crate never depends on the windowing crate: the application creates the `wgpu::Surface` on
//! the instance from [`create_instance`] and hands it to [`Gpu::new`] and [`SurfaceState`].
//! [`Gpu::headless`] needs no surface at all `[AC-WP04-1]`.

use std::fmt;

use crate::error::{DeviceEvents, GpuError};
use crate::lifecycle::LifecycleGuard;

/// 512 MiB: the storage-buffer binding size `[GPU-02]` asks for.
pub const WANTED_STORAGE_BINDING_SIZE: u64 = 512 << 20;
/// 1 GiB: the buffer size `[GPU-02]` asks for.
pub const WANTED_BUFFER_SIZE: u64 = 1 << 30;
/// Storage buffers per shader stage `[GPU-02]` asks for.
pub const WANTED_STORAGE_BUFFERS_PER_STAGE: u32 = 12;
/// Texture side that makes streaming-free mode possible `[GPU-02]`.
pub const WANTED_TEXTURE_DIMENSION: u32 = 16384;
/// Most binding-array elements per stage requested when texture binding arrays are on.
pub const WANTED_BINDING_ARRAY_ELEMENTS: u32 = 1024;

/// An optional GPU feature with a mandatory fallback (spec 14.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// 2D textures of 16384 texels a side.
    Textures16k,
    /// `f16` in shaders.
    ShaderF16,
    /// Subgroup operations in compute shaders.
    Subgroups,
    /// Timestamp queries around passes.
    TimestampQueries,
    /// Mesh shaders (experimental in wgpu 30).
    MeshShaders,
    /// Arrays of texture bindings ("bindless").
    TextureBindingArrays,
    /// Linear filtering of 32-bit float textures.
    Float32Filterable,
    /// Storage-buffer bindings of 512 MiB and buffers of 1 GiB.
    LargeStorageBuffers,
    /// BC block-compressed textures.
    TextureCompressionBc,
}

impl Capability {
    /// Every capability, in the order of the report.
    pub const ALL: [Self; 9] = [
        Self::Textures16k,
        Self::ShaderF16,
        Self::Subgroups,
        Self::TimestampQueries,
        Self::MeshShaders,
        Self::TextureBindingArrays,
        Self::Float32Filterable,
        Self::LargeStorageBuffers,
        Self::TextureCompressionBc,
    ];

    /// The stable name used in reports and in [`CapabilityOverrides::parse`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::Textures16k => "textures-16k",
            Self::ShaderF16 => "shader-f16",
            Self::Subgroups => "subgroups",
            Self::TimestampQueries => "timestamp-queries",
            Self::MeshShaders => "mesh-shaders",
            Self::TextureBindingArrays => "texture-binding-arrays",
            Self::Float32Filterable => "float32-filterable",
            Self::LargeStorageBuffers => "large-storage-buffers",
            Self::TextureCompressionBc => "texture-compression-bc",
        }
    }

    /// What the application does instead when the capability is missing (spec 14.5).
    pub const fn fallback(self) -> &'static str {
        match self {
            Self::Textures16k => "8K tiles + streaming",
            Self::ShaderF16 => "f32 math, packed storage via manual unpack",
            Self::Subgroups => "workgroup-shared-memory reductions",
            Self::TimestampQueries => "CPU frame timing + pass counters",
            Self::MeshShaders => "compute raster + visibility buffer for hair; indexed draws for meshes",
            Self::TextureBindingArrays => "texture arrays/atlases",
            Self::Float32Filterable => "f16 targets and manual filtering",
            Self::LargeStorageBuffers => "chunked buffers and multi-dispatch",
            Self::TextureCompressionBc => "uncompressed textures",
        }
    }

    /// The capability's bit in a [`CapabilitySet`].
    const fn bit(self) -> u16 {
        1 << self as u16
    }

    /// The wgpu feature that switches the capability on, for the ones that are features (the others
    /// are limits).
    const fn feature(self) -> Option<wgpu::Features> {
        match self {
            Self::ShaderF16 => Some(wgpu::Features::SHADER_F16),
            Self::Subgroups => Some(wgpu::Features::SUBGROUP),
            Self::TimestampQueries => Some(wgpu::Features::TIMESTAMP_QUERY),
            Self::MeshShaders => Some(wgpu::Features::EXPERIMENTAL_MESH_SHADER),
            Self::TextureBindingArrays => Some(wgpu::Features::TEXTURE_BINDING_ARRAY),
            Self::Float32Filterable => Some(wgpu::Features::FLOAT32_FILTERABLE),
            Self::TextureCompressionBc => Some(wgpu::Features::TEXTURE_COMPRESSION_BC),
            Self::Textures16k | Self::LargeStorageBuffers => None,
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`Capability`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CapabilitySet(u16);

impl CapabilitySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);
    /// Every capability.
    pub const ALL: Self = Self((1 << Capability::ALL.len()) - 1);

    /// True when the capability is in the set.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// The set with one capability added.
    #[must_use]
    pub const fn with(self, capability: Capability) -> Self {
        Self(self.0 | capability.bit())
    }

    /// The set with one capability removed.
    #[must_use]
    pub const fn without(self, capability: Capability) -> Self {
        Self(self.0 & !capability.bit())
    }

    /// The capabilities in `self` that are not in `other`.
    #[must_use]
    pub const fn minus(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Number of capabilities in the set.
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// True when the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The debug mask that forces capabilities off, so every fallback of spec 14.5 can be exercised on a
/// machine that has the feature `[AC-WP04-1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CapabilityOverrides {
    /// Capabilities treated as missing whatever the adapter offers.
    pub force_off: CapabilitySet,
}

impl CapabilityOverrides {
    /// Nothing forced: the adapter decides.
    pub const NONE: Self = Self { force_off: CapabilitySet::EMPTY };
    /// Every capability forced off: the pure baseline.
    pub const ALL_OFF: Self = Self { force_off: CapabilitySet::ALL };

    /// The overrides with one more capability forced off.
    #[must_use]
    pub const fn disable(self, capability: Capability) -> Self {
        Self { force_off: self.force_off.with(capability) }
    }

    /// Reads a comma-separated list of capability names (`"shader-f16,mesh-shaders"`), or `"all"`.
    /// Spaces around names and empty items are ignored.
    pub fn parse(text: &str) -> Result<Self, GpuError> {
        let mut overrides = Self::NONE;
        for item in text.split(',').map(str::trim).filter(|item| !item.is_empty()) {
            if item == "all" {
                overrides = Self::ALL_OFF;
            } else if let Some(capability) = Capability::ALL.into_iter().find(|c| c.name() == item) {
                overrides = overrides.disable(capability);
            } else {
                return Err(GpuError::Invalid { what: "unknown capability name in the override list" });
            }
        }
        Ok(overrides)
    }
}

/// What the adapter offers and what the device was created with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// Offered by the adapter.
    detected: CapabilitySet,
    /// Switched on for the device: the only set code should branch on.
    enabled: CapabilitySet,
    /// Offered but switched off by [`CapabilityOverrides`].
    forced_off: CapabilitySet,
}

impl Capabilities {
    /// True when the capability is usable on the device. When false, use [`Capability::fallback`].
    pub const fn has(self, capability: Capability) -> bool {
        self.enabled.contains(capability)
    }

    /// True when the adapter offers the capability, usable or not.
    pub const fn detected(self, capability: Capability) -> bool {
        self.detected.contains(capability)
    }

    /// True when the adapter offers the capability and the debug mask switched it off.
    pub const fn forced_off(self, capability: Capability) -> bool {
        self.forced_off.contains(capability)
    }

    /// The usable capabilities.
    pub const fn enabled(self) -> CapabilitySet {
        self.enabled
    }

    /// One word for the report.
    fn state(self, capability: Capability) -> &'static str {
        if self.has(capability) {
            "on"
        } else if self.forced_off(capability) {
            "forced off"
        } else if self.detected(capability) {
            "not enabled"
        } else {
            "missing"
        }
    }
}

/// The limits tier the device runs on `[GPU-02]`. This is about what the device can hold, not how
/// fast it is: the performance tiers L/M/H of spec 14.1 are measured separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// wgpu's default limits: 8K tiles with streaming, chunked buffers, reduced quality.
    Baseline,
    /// The raised buffer limits of `[GPU-02]` with 8K textures: tiles and streaming for 16K sets.
    Standard,
    /// Every limit of `[GPU-02]` including 16K textures: streaming-free mode is possible.
    Full,
}

impl Tier {
    /// The tier's name in reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Standard => "standard",
            Self::Full => "full",
        }
    }

    /// What the tier means for quality.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Baseline => "default limits: 8K tiles + streaming, chunked buffers, reduced quality",
            Self::Standard => "raised buffer limits, 8K textures: tiles + streaming for 16K texture sets",
            Self::Full => "all requested limits, 16K textures: streaming-free mode available",
        }
    }

    /// Derives the tier from the usable capabilities and the granted limits.
    pub fn select(capabilities: Capabilities, limits: &wgpu::Limits) -> Self {
        let buffers = capabilities.has(Capability::LargeStorageBuffers)
            && limits.max_storage_buffers_per_shader_stage >= WANTED_STORAGE_BUFFERS_PER_STAGE;
        if buffers && capabilities.has(Capability::Textures16k) {
            Self::Full
        } else if buffers {
            Self::Standard
        } else {
            Self::Baseline
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a device is asked for.
#[derive(Debug, Clone)]
pub struct DeviceRequest {
    /// The device's debug label.
    pub label: &'static str,
    /// Which adapter to prefer when there are several. `WGPU_POWER_PREF` overrides it.
    pub power_preference: wgpu::PowerPreference,
    /// Capabilities to treat as missing.
    pub overrides: CapabilityOverrides,
    /// Features outside the capability model to switch on when the adapter has them (never an
    /// error when it has not: check [`Gpu::features`]).
    pub extra_features: wgpu::Features,
    /// The token that allows experimental wgpu features. Mesh shaders are enabled only with an
    /// enabled token, which only code that may use `unsafe` can make.
    pub experimental: wgpu::ExperimentalFeatures,
    /// wgpu's allocation strategy.
    pub memory_hints: wgpu::MemoryHints,
    /// Ask for a software adapter.
    pub force_fallback_adapter: bool,
}

impl Default for DeviceRequest {
    fn default() -> Self {
        Self {
            label: "humanart",
            power_preference: wgpu::PowerPreference::HighPerformance,
            overrides: CapabilityOverrides::NONE,
            extra_features: wgpu::Features::empty(),
            experimental: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            force_fallback_adapter: false,
        }
    }
}

impl DeviceRequest {
    /// The reference application's own choices: the low-power adapter and small allocation blocks.
    /// For tools that draw only UI.
    pub fn reference_ui() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::LowPower,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Self::default()
        }
    }

    /// The request with a capability mask.
    #[must_use]
    pub fn with_overrides(mut self, overrides: CapabilityOverrides) -> Self {
        self.overrides = overrides;
        self
    }
}

/// What will be asked of the adapter, and what it means: the outcome of [`plan_device`].
#[derive(Debug, Clone)]
pub struct DevicePlan {
    /// Features to require.
    pub features: wgpu::Features,
    /// Limits to require: never above the adapter's.
    pub limits: wgpu::Limits,
    /// The capability model's verdict.
    pub capabilities: Capabilities,
    /// The limits tier.
    pub tier: Tier,
}

/// Decides features and limits from what an adapter offers `[GPU-02]` `[PLAT-05]`.
///
/// Pure: tests feed it synthetic adapters to force every capability on and off. The limits start at
/// wgpu's defaults (the baseline the application must run on) and only the `[GPU-02]` fields are
/// raised, each clamped to the adapter. An adapter below the defaults, or without compute shaders,
/// is refused with [`GpuError::BaselineNotMet`].
pub fn plan_device(
    adapter_features: wgpu::Features,
    adapter_limits: &wgpu::Limits,
    downlevel: wgpu::DownlevelFlags,
    request: &DeviceRequest,
) -> Result<DevicePlan, GpuError> {
    if !downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
        return Err(GpuError::BaselineNotMet { what: "compute shaders", required: 1, available: 0 });
    }
    let mut limits = wgpu::Limits::defaults();
    let mut below = None;
    limits.check_limits_with_fail_fn(adapter_limits, true, |what, required, available| {
        below = Some(GpuError::BaselineNotMet { what, required, available });
    });
    if let Some(error) = below {
        return Err(error);
    }

    let mut detected = CapabilitySet::EMPTY;
    for capability in Capability::ALL {
        let offered = match capability {
            Capability::Textures16k => adapter_limits.max_texture_dimension_2d >= WANTED_TEXTURE_DIMENSION,
            Capability::LargeStorageBuffers => {
                adapter_limits.max_storage_buffer_binding_size >= WANTED_STORAGE_BINDING_SIZE
                    && adapter_limits.max_buffer_size >= WANTED_BUFFER_SIZE
            }
            Capability::TextureBindingArrays => {
                adapter_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
                    && adapter_limits.max_binding_array_elements_per_shader_stage > 0
            }
            _ => capability.feature().is_some_and(|feature| adapter_features.contains(feature)),
        };
        if offered {
            detected = detected.with(capability);
        }
    }
    let forced_off = CapabilitySet(detected.0 & request.overrides.force_off.0);
    let mut enabled = detected.minus(forced_off);
    if !request.experimental.is_enabled() {
        enabled = enabled.without(Capability::MeshShaders);
    }
    let capabilities = Capabilities { detected, enabled, forced_off };

    // Extras are best effort; the capability features are exactly the enabled set, so a forced-off
    // capability cannot come back through the extras.
    let mut features = request.extra_features & adapter_features;
    for capability in Capability::ALL {
        if let Some(feature) = capability.feature() {
            features.set(feature, capabilities.has(capability));
        }
    }
    if !capabilities.has(Capability::TimestampQueries) {
        features
            .remove(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    }
    if !request.experimental.is_enabled() {
        features.remove(wgpu::Features::all_experimental_mask());
    }

    if capabilities.has(Capability::Textures16k) {
        limits.max_texture_dimension_2d = WANTED_TEXTURE_DIMENSION;
    }
    limits.max_storage_buffers_per_shader_stage = adapter_limits
        .max_storage_buffers_per_shader_stage
        .min(WANTED_STORAGE_BUFFERS_PER_STAGE)
        .max(limits.max_storage_buffers_per_shader_stage);
    if !request.overrides.force_off.contains(Capability::LargeStorageBuffers) {
        // Also below the capability's threshold the adapter's larger-than-default value is taken:
        // "otherwise the lower value" [GPU-02].
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(WANTED_STORAGE_BINDING_SIZE)
            .max(limits.max_storage_buffer_binding_size);
        limits.max_buffer_size = adapter_limits.max_buffer_size.min(WANTED_BUFFER_SIZE).max(limits.max_buffer_size);
    }
    if capabilities.has(Capability::TextureBindingArrays) {
        limits.max_binding_array_elements_per_shader_stage =
            adapter_limits.max_binding_array_elements_per_shader_stage.min(WANTED_BINDING_ARRAY_ELEMENTS);
    }
    if capabilities.has(Capability::MeshShaders) {
        limits.max_task_workgroup_total_count = adapter_limits.max_task_workgroup_total_count;
        limits.max_task_workgroups_per_dimension = adapter_limits.max_task_workgroups_per_dimension;
        limits.max_mesh_workgroup_total_count = adapter_limits.max_mesh_workgroup_total_count;
        limits.max_mesh_workgroups_per_dimension = adapter_limits.max_mesh_workgroups_per_dimension;
        limits.max_task_invocations_per_workgroup = adapter_limits.max_task_invocations_per_workgroup;
        limits.max_task_invocations_per_dimension = adapter_limits.max_task_invocations_per_dimension;
        limits.max_mesh_invocations_per_workgroup = adapter_limits.max_mesh_invocations_per_workgroup;
        limits.max_mesh_invocations_per_dimension = adapter_limits.max_mesh_invocations_per_dimension;
        limits.max_task_payload_size = adapter_limits.max_task_payload_size;
        limits.max_mesh_output_vertices = adapter_limits.max_mesh_output_vertices;
        limits.max_mesh_output_primitives = adapter_limits.max_mesh_output_primitives;
        limits.max_mesh_output_layers = adapter_limits.max_mesh_output_layers;
    }
    let tier = Tier::select(capabilities, &limits);
    Ok(DevicePlan { features, limits, capabilities, tier })
}

/// Creates the wgpu instance the reference's way: every compiled-in backend, overridable through
/// the `WGPU_BACKEND` environment variable. Pass the window system's display handle when there
/// will be a window, `None` for headless use. Created under the process-wide [`LifecycleGuard`].
pub fn create_instance(display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>) -> wgpu::Instance {
    create_instance_from(match display {
        Some(display) => wgpu::InstanceDescriptor::new_with_display_handle_from_env(display),
        None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
    })
}

/// Creates an instance from `descriptor` under the process-wide [`LifecycleGuard`] (the GL backend
/// creates driver contexts already here).
fn create_instance_from(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
    let _lifecycle = LifecycleGuard::acquire();
    wgpu::Instance::new(descriptor)
}

/// The headless instance on the primary backends only (Vulkan, Metal, DX12, WebGPU), unless the
/// `WGPU_BACKEND` environment variable names others. Without the GL backend no hidden window and
/// no GL context are created, which keeps one more driver out of device creation (see
/// [`crate::lifecycle`]).
#[cfg(not(target_arch = "wasm32"))]
fn create_primary_headless_instance() -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::PRIMARY;
    create_instance_from(descriptor.with_env())
}

/// A device with its queue, the adapter it came from, its capabilities and its event queue.
///
/// `device` and `queue` are public and cheap to clone (they are handles).
#[derive(Debug)]
pub struct Gpu {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The device's queue.
    pub queue: wgpu::Queue,
    /// The instance, kept to create the replacement device after a loss.
    instance: wgpu::Instance,
    /// The adapter, kept for surface capabilities.
    adapter: wgpu::Adapter,
    /// The adapter's description, read once.
    info: wgpu::AdapterInfo,
    /// What was asked for, to ask again after a loss.
    request: DeviceRequest,
    /// What was granted.
    plan: DevicePlan,
    /// Uncaptured errors and device loss.
    events: DeviceEvents,
    /// Set by `Drop` and dropped last: the device, queue, adapter and instance above are released
    /// while the process-wide lifecycle lock is held (see [`crate::lifecycle`]).
    lifecycle: Option<LifecycleGuard>,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Fields drop after this returns, in declaration order, so the guard (the last field) is
        // released after everything else. Clones of `device` and `queue` held elsewhere keep the
        // device alive past this point; whoever drops the last one should hold the guard too.
        self.lifecycle = Some(LifecycleGuard::acquire());
    }
}

impl Gpu {
    /// Creates a device on `instance`. `compatible_surface` makes the adapter one that can present
    /// to the application's window. The adapter and device are created, and on failure dropped,
    /// under the process-wide [`LifecycleGuard`]; the caller must not hold one.
    pub async fn new(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        request: &DeviceRequest,
    ) -> Result<Self, GpuError> {
        // Declared before the adapter and the instance, so on every early return they drop before
        // the guard does. On native, wgpu's
        // creation futures are ready at once, so the guard is never held across a real suspension.
        let lifecycle = LifecycleGuard::acquire();
        // Re-bound after the guard: arguments would otherwise drop after it.
        let instance = instance;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::from_env().unwrap_or(request.power_preference),
                force_fallback_adapter: request.force_fallback_adapter,
                compatible_surface,
                ..Default::default()
            })
            .await
            .map_err(GpuError::NoAdapter)?;
        let plan =
            plan_device(adapter.features(), &adapter.limits(), adapter.get_downlevel_capabilities().flags, request)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(request.label),
                required_features: plan.features,
                required_limits: plan.limits.clone(),
                experimental_features: request.experimental,
                memory_hints: request.memory_hints.clone(),
                ..Default::default()
            })
            .await
            .map_err(GpuError::Device)?;
        let events = DeviceEvents::new();
        events.attach(&device);
        let info = adapter.get_info();
        let gpu =
            Self { device, queue, instance, adapter, info, request: request.clone(), plan, events, lifecycle: None };
        drop(lifecycle);
        Ok(gpu)
    }

    /// Creates a device with no window and no surface: for tests, the command-line tool and
    /// servers `[AC-WP04-1]`.
    ///
    /// On native targets the primary backends (Vulkan, Metal, DX12) are tried first and every
    /// backend only when none of them has an adapter, unless `WGPU_BACKEND` chooses. Processes that
    /// only need some GPU should share [`crate::lifecycle::shared_headless`] instead.
    pub async fn headless(request: &DeviceRequest) -> Result<Self, GpuError> {
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("WGPU_BACKEND").is_none() {
            match Self::new(create_primary_headless_instance(), None, request).await {
                Err(GpuError::NoAdapter(_)) => {}
                other => return other,
            }
        }
        Self::new(create_instance(None), None, request).await
    }

    /// [`Gpu::headless`] for code that is not async. Native only: a browser cannot block.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless_blocking(request: &DeviceRequest) -> Result<Self, GpuError> {
        block_on(Self::headless(request))
    }

    /// Creates a new device with the same request on the same instance, for recovery after a
    /// device loss `[ERR-05]`. Everything made on the old device must be rebuilt
    /// (see [`crate::recovery`]).
    pub async fn recreate(&self, compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self, GpuError> {
        Self::new(self.instance.clone(), compatible_surface, &self.request).await
    }

    /// [`Gpu::recreate`] for code that is not async. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn recreate_blocking(&self, compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self, GpuError> {
        block_on(self.recreate(compatible_surface))
    }

    /// The usable capabilities.
    pub fn capabilities(&self) -> Capabilities {
        self.plan.capabilities
    }

    /// Shorthand for `capabilities().has(capability)`.
    pub fn has(&self, capability: Capability) -> bool {
        self.plan.capabilities.has(capability)
    }

    /// The limits tier the device runs on.
    pub fn tier(&self) -> Tier {
        self.plan.tier
    }

    /// The features the device was created with.
    pub fn features(&self) -> wgpu::Features {
        self.plan.features
    }

    /// The limits the device was created with.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.plan.limits
    }

    /// The adapter's name, backend and driver.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// The adapter, for surface capabilities and format features.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// The instance, to create surfaces on.
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// The request the device was created from.
    pub fn request(&self) -> &DeviceRequest {
        &self.request
    }

    /// The queue of uncaptured errors and device loss.
    pub fn events(&self) -> &DeviceEvents {
        &self.events
    }

    /// True once the device has been lost. One atomic load.
    #[inline]
    pub fn is_lost(&self) -> bool {
        self.events.is_lost()
    }

    /// The human-readable report of adapter, tier, capabilities with their fallbacks, and the
    /// granted `[GPU-02]` limits. The application logs it at startup and the benchmarks record it.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(1024);
        // Writing to a `String` cannot fail.
        let _ = self.write_report(&mut out);
        out
    }

    /// Writes [`Gpu::report`] into any formatter.
    pub fn write_report(&self, out: &mut impl fmt::Write) -> fmt::Result {
        let info = &self.info;
        writeln!(out, "adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type)?;
        writeln!(out, "driver: {} {}", info.driver, info.driver_info)?;
        writeln!(out, "tier: {} ({})", self.plan.tier, self.plan.tier.description())?;
        for capability in Capability::ALL {
            let state = self.plan.capabilities.state(capability);
            if self.plan.capabilities.has(capability) {
                writeln!(out, "capability {capability}: {state}")?;
            } else {
                writeln!(out, "capability {capability}: {state}; fallback: {}", capability.fallback())?;
            }
        }
        let limits = &self.plan.limits;
        writeln!(out, "limit max_texture_dimension_2d: {}", limits.max_texture_dimension_2d)?;
        writeln!(out, "limit max_storage_buffer_binding_size: {}", limits.max_storage_buffer_binding_size)?;
        writeln!(out, "limit max_buffer_size: {}", limits.max_buffer_size)?;
        writeln!(out, "limit max_storage_buffers_per_shader_stage: {}", limits.max_storage_buffers_per_shader_stage)?;
        writeln!(out, "limit max_compute_workgroup_storage_size: {}", limits.max_compute_workgroup_storage_size)?;
        writeln!(out, "limit max_bind_groups: {}", limits.max_bind_groups)
    }
}

/// Minimal executor for wgpu's initialisation futures (the reference's `block_on`).
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: Future>(future: F) -> F::Output {
    /// Wakes the blocked thread.
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::sync::Arc::new(Unpark(std::thread::current())).into();
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::park();
    }
}

/// Picks the surface format the reference's way: the first that is not sRGB-encoded, else the
/// first, else [`GpuError::NoSurfaceFormat`].
pub fn choose_surface_format(formats: &[wgpu::TextureFormat]) -> Result<wgpu::TextureFormat, GpuError> {
    formats
        .iter()
        .copied()
        .find(|format| !format.is_srgb())
        .or_else(|| formats.first().copied())
        .ok_or(GpuError::NoSurfaceFormat)
}

/// Builds the surface configuration the reference's way from the surface's capabilities: render
/// attachment only, `AutoVsync`, frame latency 2, the first alpha mode, sizes at least 1.
pub fn surface_configuration(
    capabilities: &wgpu::SurfaceCapabilities,
    width: u32,
    height: u32,
) -> Result<wgpu::SurfaceConfiguration, GpuError> {
    Ok(wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: choose_surface_format(&capabilities.formats)?,
        color_space: wgpu::SurfaceColorSpace::Auto,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: capabilities.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
        view_formats: Vec::new(),
    })
}

/// What to do with a frame, after asking the surface for its texture.
#[derive(Debug)]
pub enum FrameAcquire {
    /// Draw into this texture, then `queue.present` it.
    Draw(wgpu::SurfaceTexture),
    /// The surface was reconfigured: ask for a redraw at once.
    Retry,
    /// Nothing to draw into now (hidden, occluded, timed out, or a second failure in a row): wait
    /// as for an idle frame.
    Skip,
}

/// What the surface answered, without the texture: the input of the retry rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceAnswer {
    /// A texture, perhaps suboptimal.
    Texture,
    /// Outdated or lost: reconfigure.
    Reconfigure,
    /// Timeout, occluded or a validation error.
    Unavailable,
}

/// A window surface's configuration with the reference's resize and retry rules (section 2 of the
/// reference conventions): a resize only stores the size and the surface is configured once at the
/// next frame; a zero size means hidden; a surface that keeps refusing cannot spin the loop.
#[derive(Debug)]
pub struct SurfaceState {
    /// The current configuration.
    config: wgpu::SurfaceConfiguration,
    /// A resize is waiting to be applied.
    resized: bool,
    /// The window has no area.
    hidden: bool,
    /// The previous frame was a retry.
    retried: bool,
}

impl SurfaceState {
    /// Configures `surface` for `gpu` at the given size in physical pixels.
    pub fn new(surface: &wgpu::Surface<'_>, gpu: &Gpu, width: u32, height: u32) -> Result<Self, GpuError> {
        let config = surface_configuration(&surface.get_capabilities(&gpu.adapter), width, height)?;
        let hidden = width == 0 || height == 0;
        if !hidden {
            surface.configure(&gpu.device, &config);
        }
        Ok(Self { config, resized: hidden, hidden, retried: false })
    }

    /// The current configuration (format, size).
    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    /// The surface's texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// True while the window has no area.
    pub fn is_hidden(&self) -> bool {
        self.hidden
    }

    /// Records a new size in physical pixels. The surface is configured at the next
    /// [`SurfaceState::acquire`], however many sizes passed by.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.hidden = width == 0 || height == 0;
        if !self.hidden && (width != self.config.width || height != self.config.height || self.resized) {
            self.config.width = width;
            self.config.height = height;
            self.resized = true;
        }
    }

    /// Changes the present mode; applied at the next frame.
    pub fn set_present_mode(&mut self, mode: wgpu::PresentMode) {
        if self.config.present_mode != mode {
            self.config.present_mode = mode;
            self.resized = true;
        }
    }

    /// Tells whether a frame may be attempted: not while the window has no area. The pending
    /// resize itself is applied by [`SurfaceState::acquire`].
    fn begin(&mut self) -> bool {
        !self.hidden
    }

    /// The retry rule: reconfigure answers give one immediate retry, then skip.
    fn decide(&mut self, answer: SurfaceAnswer) -> SurfaceAnswer {
        match answer {
            SurfaceAnswer::Texture => {
                self.retried = false;
                SurfaceAnswer::Texture
            }
            SurfaceAnswer::Reconfigure => {
                if std::mem::replace(&mut self.retried, true) {
                    SurfaceAnswer::Unavailable
                } else {
                    SurfaceAnswer::Reconfigure
                }
            }
            SurfaceAnswer::Unavailable => SurfaceAnswer::Unavailable,
        }
    }

    /// Asks the surface for this frame's texture, reconfiguring as needed.
    pub fn acquire(&mut self, surface: &wgpu::Surface<'_>, device: &wgpu::Device) -> FrameAcquire {
        if !self.begin() {
            return FrameAcquire::Skip;
        }
        if std::mem::take(&mut self.resized) {
            surface.configure(device, &self.config);
        }
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => {
                self.decide(SurfaceAnswer::Texture);
                FrameAcquire::Draw(frame)
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Draw this one, reconfigure for the next.
                surface.configure(device, &self.config);
                self.decide(SurfaceAnswer::Texture);
                FrameAcquire::Draw(frame)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(device, &self.config);
                match self.decide(SurfaceAnswer::Reconfigure) {
                    SurfaceAnswer::Reconfigure => FrameAcquire::Retry,
                    _ => FrameAcquire::Skip,
                }
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => FrameAcquire::Skip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    /// An adapter that offers everything the capability model knows.
    fn rich_adapter() -> (wgpu::Features, wgpu::Limits) {
        let mut limits = wgpu::Limits::defaults();
        limits.max_texture_dimension_2d = 32768;
        limits.max_storage_buffer_binding_size = 4 << 30;
        limits.max_buffer_size = 16 << 30;
        limits.max_storage_buffers_per_shader_stage = 64;
        limits.max_binding_array_elements_per_shader_stage = 500_000;
        limits.max_mesh_output_vertices = 256;
        (wgpu::Features::all(), limits)
    }

    fn plan(features: wgpu::Features, limits: &wgpu::Limits, request: &DeviceRequest) -> DevicePlan {
        plan_device(features, limits, wgpu::DownlevelFlags::all(), request).unwrap()
    }

    #[test]
    fn ac_wp04_1_every_capability_can_be_forced_off_and_is_on_otherwise() {
        let (features, limits) = rich_adapter();
        for capability in Capability::ALL {
            // On: the adapter offers it and nothing forces it off. Mesh shaders are experimental
            // in wgpu and stay off without the token, which only unsafe code can make.
            let on = plan(features, &limits, &DeviceRequest::default());
            assert!(on.capabilities.detected(capability), "{capability} detected");
            assert_eq!(on.capabilities.has(capability), capability != Capability::MeshShaders, "{capability} on");
            assert!(!on.capabilities.forced_off(capability));

            // Off: the debug mask removes exactly this one.
            let overrides = CapabilityOverrides::NONE.disable(capability);
            let off = plan(features, &limits, &DeviceRequest::default().with_overrides(overrides));
            assert!(off.capabilities.detected(capability));
            assert!(!off.capabilities.has(capability), "{capability} forced off");
            assert!(off.capabilities.forced_off(capability));
            if let Some(feature) = capability.feature() {
                assert!(!off.features.contains(feature), "{capability}: its wgpu feature is not requested");
            }
            for other in Capability::ALL {
                if other != capability {
                    assert_eq!(off.capabilities.has(other), on.capabilities.has(other), "{other} untouched");
                }
            }
        }
    }

    #[test]
    fn ac_wp04_1_forced_off_capabilities_lower_limits_and_tier() {
        let (features, limits) = rich_adapter();
        let full = plan(features, &limits, &DeviceRequest::default());
        assert_eq!(full.tier, Tier::Full);
        assert_eq!(full.limits.max_texture_dimension_2d, 16384);
        assert_eq!(full.limits.max_storage_buffer_binding_size, 512 << 20);
        assert_eq!(full.limits.max_buffer_size, 1 << 30);
        assert_eq!(full.limits.max_storage_buffers_per_shader_stage, 12);
        assert_eq!(full.limits.max_binding_array_elements_per_shader_stage, WANTED_BINDING_ARRAY_ELEMENTS);
        assert_eq!(full.limits.max_mesh_output_vertices, 0, "mesh limits stay off without the token");

        let no_16k = CapabilityOverrides::NONE.disable(Capability::Textures16k);
        let standard = plan(features, &limits, &DeviceRequest::default().with_overrides(no_16k));
        assert_eq!(standard.tier, Tier::Standard);
        assert_eq!(standard.limits.max_texture_dimension_2d, 8192);

        let no_large = CapabilityOverrides::NONE.disable(Capability::LargeStorageBuffers);
        let baseline = plan(features, &limits, &DeviceRequest::default().with_overrides(no_large));
        assert_eq!(baseline.tier, Tier::Baseline);
        assert_eq!(baseline.limits.max_storage_buffer_binding_size, 128 << 20);
        assert_eq!(baseline.limits.max_buffer_size, 256 << 20);

        let all_off = plan(features, &limits, &DeviceRequest::default().with_overrides(CapabilityOverrides::ALL_OFF));
        assert!(all_off.capabilities.enabled().is_empty());
        assert_eq!(all_off.tier, Tier::Baseline);
        assert_eq!(all_off.limits.max_binding_array_elements_per_shader_stage, 0);
        assert_eq!(all_off.capabilities.forced_off, CapabilitySet::ALL);
    }

    #[test]
    fn gpu_02_default_adapter_runs_at_baseline_and_partial_limits_are_taken() {
        // An adapter with exactly the default limits and no optional feature.
        let defaults = wgpu::Limits::defaults();
        let poor = plan(wgpu::Features::empty(), &defaults, &DeviceRequest::default());
        assert!(poor.capabilities.enabled().is_empty());
        assert!(poor.features.is_empty());
        assert_eq!(poor.tier, Tier::Baseline);
        assert!(poor.limits.check_limits(&defaults), "never asks for more than the adapter has");

        // Between the default and the wanted value: the adapter's value is requested.
        let mut middle = wgpu::Limits::defaults();
        middle.max_storage_buffer_binding_size = 256 << 20;
        middle.max_buffer_size = 512 << 20;
        middle.max_storage_buffers_per_shader_stage = 10;
        let plan = plan(wgpu::Features::empty(), &middle, &DeviceRequest::default());
        assert!(!plan.capabilities.has(Capability::LargeStorageBuffers));
        assert_eq!(plan.limits.max_storage_buffer_binding_size, 256 << 20);
        assert_eq!(plan.limits.max_buffer_size, 512 << 20);
        assert_eq!(plan.limits.max_storage_buffers_per_shader_stage, 10);
        assert_eq!(plan.tier, Tier::Baseline);
    }

    #[test]
    fn plat_05_an_adapter_below_the_baseline_is_refused_with_the_reason() {
        let mut limits = wgpu::Limits::defaults();
        limits.max_texture_dimension_2d = 4096;
        let error =
            plan_device(wgpu::Features::empty(), &limits, wgpu::DownlevelFlags::all(), &DeviceRequest::default())
                .unwrap_err();
        match error {
            GpuError::BaselineNotMet { what, required, available } => {
                assert_eq!((what, required, available), ("max_texture_dimension_2d", 8192, 4096));
            }
            other => panic!("unexpected error {other}"),
        }
        let no_compute = plan_device(
            wgpu::Features::empty(),
            &wgpu::Limits::defaults(),
            wgpu::DownlevelFlags::empty(),
            &DeviceRequest::default(),
        );
        assert!(matches!(no_compute, Err(GpuError::BaselineNotMet { what: "compute shaders", .. })));
    }

    #[test]
    fn extras_are_best_effort_and_cannot_undo_an_override() {
        let (features, limits) = rich_adapter();
        let request = DeviceRequest {
            extra_features: wgpu::Features::SHADER_F16
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            overrides: CapabilityOverrides::NONE.disable(Capability::ShaderF16).disable(Capability::TimestampQueries),
            ..DeviceRequest::default()
        };
        let plan = plan(features, &limits, &request);
        assert!(plan.features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8));
        assert!(!plan.features.contains(wgpu::Features::SHADER_F16));
        assert!(!plan.features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS));
        assert!(!plan.features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY), "no experimental without token");
        // An adapter without the extra: no error, just absent.
        let none = super::plan_device(wgpu::Features::empty(), &limits, wgpu::DownlevelFlags::all(), &request).unwrap();
        assert!(none.features.is_empty());
    }

    #[test]
    fn overrides_parse_names_and_reject_unknown_ones() {
        let parsed = CapabilityOverrides::parse(" shader-f16, mesh-shaders ,,").unwrap();
        assert_eq!(parsed, CapabilityOverrides::NONE.disable(Capability::ShaderF16).disable(Capability::MeshShaders));
        assert_eq!(CapabilityOverrides::parse("").unwrap(), CapabilityOverrides::NONE);
        assert_eq!(CapabilityOverrides::parse("all").unwrap(), CapabilityOverrides::ALL_OFF);
        assert!(CapabilityOverrides::parse("shader-f17").is_err());
        for capability in Capability::ALL {
            assert_eq!(CapabilityOverrides::parse(capability.name()).unwrap().force_off.len(), 1);
            assert!(!capability.fallback().is_empty());
        }
        assert_eq!(CapabilitySet::ALL.len(), 9);
    }

    #[test]
    fn ac_wp04_1_headless_device_is_created_and_reports_its_tier() {
        let gpu = testing::gpu();
        let report = gpu.report();
        assert!(report.contains(&format!("adapter: {}", gpu.adapter_info().name)));
        assert!(report.contains(&format!("tier: {}", gpu.tier())));
        for capability in Capability::ALL {
            assert!(report.contains(&format!("capability {capability}: ")), "{report}");
        }
        // What the plan says is what the device has.
        assert_eq!(gpu.device.features(), gpu.features());
        assert_eq!(gpu.device.limits().max_texture_dimension_2d, gpu.limits().max_texture_dimension_2d);
        assert_eq!(gpu.device.limits().max_buffer_size, gpu.limits().max_buffer_size);
        assert_eq!(gpu.tier(), Tier::select(gpu.capabilities(), gpu.limits()));
        assert!(!gpu.is_lost());
        println!("{report}");
    }

    #[test]
    fn ac_wp04_1_a_real_device_with_every_capability_forced_off_is_the_baseline() {
        let request = DeviceRequest::default().with_overrides(CapabilityOverrides::ALL_OFF);
        let gpu = Gpu::headless_blocking(&request).unwrap();
        assert_eq!(gpu.tier(), Tier::Baseline);
        assert!(gpu.capabilities().enabled().is_empty());
        assert!(gpu.device.features().is_empty(), "no optional feature on the device: {:?}", gpu.device.features());
        assert_eq!(gpu.device.limits().max_texture_dimension_2d, 8192);
        assert_eq!(gpu.device.limits().max_storage_buffer_binding_size, 128 << 20);
        let shared = testing::gpu();
        for capability in Capability::ALL {
            assert_eq!(gpu.capabilities().detected(capability), shared.capabilities().detected(capability));
            assert_eq!(gpu.capabilities().forced_off(capability), shared.capabilities().detected(capability));
            assert!(gpu.report().contains(capability.fallback()));
        }
        // The baseline device works: a buffer round-trips through the queue.
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("baseline"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, &[7; 16]);
        gpu.queue.submit([]);
        assert_eq!(testing::read_mappable(&gpu, &buffer), vec![7; 16]);
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn surface_format_prefers_non_srgb_like_the_reference() {
        use wgpu::TextureFormat as F;
        assert_eq!(choose_surface_format(&[F::Bgra8UnormSrgb, F::Bgra8Unorm]).unwrap(), F::Bgra8Unorm);
        assert_eq!(choose_surface_format(&[F::Bgra8UnormSrgb]).unwrap(), F::Bgra8UnormSrgb);
        assert!(matches!(choose_surface_format(&[]), Err(GpuError::NoSurfaceFormat)));
        let capabilities = wgpu::SurfaceCapabilities {
            formats: vec![F::Rgba8UnormSrgb, F::Rgba8Unorm],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            ..Default::default()
        };
        let config = surface_configuration(&capabilities, 0, 720).unwrap();
        assert_eq!((config.width, config.height, config.format), (1, 720, F::Rgba8Unorm));
        assert_eq!(config.present_mode, wgpu::PresentMode::AutoVsync);
        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn surface_state_defers_resizes_and_retries_once() {
        let capabilities =
            wgpu::SurfaceCapabilities { formats: vec![wgpu::TextureFormat::Bgra8Unorm], ..Default::default() };
        let config = surface_configuration(&capabilities, 800, 600).unwrap();
        let mut state = SurfaceState { config, resized: false, hidden: false, retried: false };
        state.resize(800, 600);
        assert!(!state.resized, "the same size is not a resize");
        state.resize(1024, 768);
        state.resize(1280, 720);
        assert!(state.resized);
        assert_eq!((state.config().width, state.config().height), (1280, 720));
        state.resize(0, 720);
        assert!(state.is_hidden() && !state.begin(), "a zero size hides and is never configured");
        assert_eq!(state.config().width, 1280);
        state.resize(1280, 720);
        assert!(state.begin());

        // Outdated twice in a row: one retry, then a skip; a texture resets the rule.
        assert_eq!(state.decide(SurfaceAnswer::Reconfigure), SurfaceAnswer::Reconfigure);
        assert_eq!(state.decide(SurfaceAnswer::Reconfigure), SurfaceAnswer::Unavailable);
        assert_eq!(state.decide(SurfaceAnswer::Texture), SurfaceAnswer::Texture);
        assert_eq!(state.decide(SurfaceAnswer::Reconfigure), SurfaceAnswer::Reconfigure);
        assert_eq!(state.decide(SurfaceAnswer::Unavailable), SurfaceAnswer::Unavailable);
        state.set_present_mode(wgpu::PresentMode::AutoNoVsync);
        assert!(state.resized);
    }

    #[test]
    fn block_on_drives_a_future_that_wakes_from_another_thread() {
        use std::sync::{Arc, Mutex};
        use std::task::{Poll, Waker};
        let shared: Arc<Mutex<(bool, Option<Waker>)>> = Arc::new(Mutex::new((false, None)));
        let other = Arc::clone(&shared);
        let thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let mut state = other.lock().unwrap();
            state.0 = true;
            if let Some(waker) = state.1.take() {
                waker.wake();
            }
        });
        let value = block_on(std::future::poll_fn(|context| {
            let mut state = shared.lock().unwrap();
            if state.0 {
                Poll::Ready(42)
            } else {
                state.1 = Some(context.waker().clone());
                Poll::Pending
            }
        }));
        assert_eq!(value, 42);
        thread.join().unwrap();
    }
}
