//! The pipeline and bind-group cache `[PERF-R4]`.
//!
//! Shader modules, bind-group layouts, pipeline layouts, render and compute pipelines and bind groups
//! are created once, stored under a key the caller chooses, and looked up by that key afterwards.
//! A lookup is one hash-map probe and never allocates, so frame code fetches its pipelines from the
//! cache instead of holding them in a dozen fields.
//!
//! The key type `K` is the caller's own (an enum of its passes and variants, say), so two different
//! objects can never collide under one hash. Every kind of object has its own map, so one key may name
//! a shader, its layout and its pipeline at the same time.
//!
//! **Pre-warming.** `[PERF-R4]` forbids pipeline creation during interaction except for a variant's
//! first use, which must be pre-warmed. The owner creates every variant it knows at startup and then
//! calls [`PipelineCache::mark_warm`]; every creation after that is counted in
//! [`CacheStats::late`], which tests and the profiler overlay assert to stay zero.
//!
//! **Device loss** `[ERR-05]`. The cache cannot re-create what it holds by itself: pipeline
//! descriptors borrow modules and layouts and are built by the owner's code. [`PipelineCache::reset`]
//! empties it and moves it to the new device; the owner then runs its pre-warm again
//! ([`crate::recovery`]).
//!
//! Creation goes through wgpu error scopes on native targets, so a shader that does not compile or a
//! descriptor wgpu rejects comes back as [`GpuError::Validation`] and nothing is cached.

use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;

use humanart_core::hash::FastMap;

use crate::device::Gpu;
use crate::error::{GpuError, guarded};

/// Most bind groups a pipeline layout made through [`PipelineCache::pipeline_layout`] may have.
/// WebGPU guarantees 4; no adapter offers more than 8.
pub const MAX_BIND_GROUPS: usize = 8;

/// The kinds of object the cache holds, for [`CacheStats::created`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachedKind {
    /// A shader module.
    Shader,
    /// A bind-group layout.
    BindGroupLayout,
    /// A pipeline layout.
    PipelineLayout,
    /// A render pipeline.
    RenderPipeline,
    /// A compute pipeline.
    ComputePipeline,
    /// A bind group.
    BindGroup,
}

impl CachedKind {
    /// Number of kinds.
    pub const COUNT: usize = 6;
    /// Every kind, in report order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Shader,
        Self::BindGroupLayout,
        Self::PipelineLayout,
        Self::RenderPipeline,
        Self::ComputePipeline,
        Self::BindGroup,
    ];

    /// The kind's name in reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shader => "shader module",
            Self::BindGroupLayout => "bind-group layout",
            Self::PipelineLayout => "pipeline layout",
            Self::RenderPipeline => "render pipeline",
            Self::ComputePipeline => "compute pipeline",
            Self::BindGroup => "bind group",
        }
    }
}

/// What the cache did so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    /// Objects created per kind, indexed by `CachedKind as usize`. Cumulative across resets.
    pub created: [u32; CachedKind::COUNT],
    /// `get_or_create` calls answered from the cache.
    pub hits: u64,
    /// Objects other than bind groups created after [`PipelineCache::mark_warm`]: should stay zero
    /// `[PERF-R4]`.
    pub late: u32,
    /// Creations wgpu refused (nothing was cached for them).
    pub failed: u32,
    /// Times the cache was emptied for a new device.
    pub resets: u32,
}

impl CacheStats {
    /// Objects created of one kind.
    pub const fn of(&self, kind: CachedKind) -> u32 {
        self.created[kind as usize]
    }

    /// Objects created of every kind.
    pub fn total_created(&self) -> u32 {
        self.created.iter().sum()
    }
}

/// The objects that pipelines are assembled from, handed to the creation closures of
/// [`PipelineCache::render_pipeline`] and [`PipelineCache::compute_pipeline`].
pub struct Parts<'a, K> {
    /// The device.
    pub device: &'a wgpu::Device,
    /// Cached shader modules.
    shaders: &'a FastMap<K, wgpu::ShaderModule>,
    /// Cached bind-group layouts.
    bind_group_layouts: &'a FastMap<K, wgpu::BindGroupLayout>,
    /// Cached pipeline layouts.
    pipeline_layouts: &'a FastMap<K, wgpu::PipelineLayout>,
}

impl<K: Hash + Eq> Parts<'_, K> {
    /// A cached shader module, or [`GpuError::Stale`] when none was created under `key`.
    pub fn shader(&self, key: &K) -> Result<&wgpu::ShaderModule, GpuError> {
        self.shaders.get(key).ok_or(GpuError::Stale { what: "shader module key" })
    }

    /// A cached bind-group layout, or [`GpuError::Stale`].
    pub fn bind_group_layout(&self, key: &K) -> Result<&wgpu::BindGroupLayout, GpuError> {
        self.bind_group_layouts.get(key).ok_or(GpuError::Stale { what: "bind-group layout key" })
    }

    /// A cached pipeline layout, or [`GpuError::Stale`].
    pub fn pipeline_layout(&self, key: &K) -> Result<&wgpu::PipelineLayout, GpuError> {
        self.pipeline_layouts.get(key).ok_or(GpuError::Stale { what: "pipeline layout key" })
    }
}

/// The cache. See the module documentation.
pub struct PipelineCache<K> {
    /// The device objects are made on.
    device: wgpu::Device,
    /// Shader modules.
    shaders: FastMap<K, wgpu::ShaderModule>,
    /// Bind-group layouts.
    bind_group_layouts: FastMap<K, wgpu::BindGroupLayout>,
    /// Pipeline layouts.
    pipeline_layouts: FastMap<K, wgpu::PipelineLayout>,
    /// Render pipelines.
    render: FastMap<K, wgpu::RenderPipeline>,
    /// Compute pipelines.
    compute: FastMap<K, wgpu::ComputePipeline>,
    /// Bind groups.
    bind_groups: FastMap<K, wgpu::BindGroup>,
    /// The counters.
    stats: CacheStats,
    /// Set by [`PipelineCache::mark_warm`].
    warm: bool,
}

impl<K> fmt::Debug for PipelineCache<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCache")
            .field("shaders", &self.shaders.len())
            .field("bind_group_layouts", &self.bind_group_layouts.len())
            .field("pipeline_layouts", &self.pipeline_layouts.len())
            .field("render", &self.render.len())
            .field("compute", &self.compute.len())
            .field("bind_groups", &self.bind_groups.len())
            .field("stats", &self.stats)
            .field("warm", &self.warm)
            .finish_non_exhaustive()
    }
}

/// Looks `key` up in `map`, or creates the value with `create` (inside error scopes) and stores it.
fn get_or_create<'m, K: Hash + Eq, V>(
    map: &'m mut FastMap<K, V>,
    stats: &mut CacheStats,
    warm: bool,
    device: &wgpu::Device,
    kind: CachedKind,
    key: K,
    create: impl FnOnce() -> Result<V, GpuError>,
) -> Result<&'m V, GpuError> {
    match map.entry(key) {
        Entry::Occupied(entry) => {
            stats.hits += 1;
            Ok(entry.into_mut())
        }
        Entry::Vacant(entry) => {
            let value = match guarded(device, kind.name(), create).and_then(|value| value) {
                Ok(value) => value,
                Err(error) => {
                    stats.failed += 1;
                    return Err(error);
                }
            };
            stats.created[kind as usize] += 1;
            // Bind groups are re-made whenever a bound resource is replaced (a resize): expected,
            // and cheap. Everything else made after the pre-warm is a hitch [PERF-R4].
            if warm && kind != CachedKind::BindGroup {
                stats.late += 1;
            }
            Ok(entry.insert(value))
        }
    }
}

impl<K: Hash + Eq> PipelineCache<K> {
    /// An empty cache on `gpu`'s device.
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            device: gpu.device.clone(),
            shaders: FastMap::default(),
            bind_group_layouts: FastMap::default(),
            pipeline_layouts: FastMap::default(),
            render: FastMap::default(),
            compute: FastMap::default(),
            bind_groups: FastMap::default(),
            stats: CacheStats::default(),
            warm: false,
        }
    }

    /// The counters.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Declares the pre-warm finished: every creation from now on counts as late `[PERF-R4]`.
    pub fn mark_warm(&mut self) {
        self.warm = true;
    }

    /// True after [`PipelineCache::mark_warm`] and until the next [`PipelineCache::reset`].
    pub fn is_warm(&self) -> bool {
        self.warm
    }

    /// The shader module compiled from WGSL `source` under `key`, compiling it at first use.
    /// `source` is read only on a miss.
    pub fn shader(&mut self, key: K, label: &str, source: &str) -> Result<&wgpu::ShaderModule, GpuError> {
        let device = &self.device;
        get_or_create(&mut self.shaders, &mut self.stats, self.warm, device, CachedKind::Shader, key, || {
            Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            }))
        })
    }

    /// The bind-group layout with `entries` under `key`, created at first use.
    pub fn bind_group_layout(
        &mut self,
        key: K,
        label: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<&wgpu::BindGroupLayout, GpuError> {
        let device = &self.device;
        get_or_create(
            &mut self.bind_group_layouts,
            &mut self.stats,
            self.warm,
            device,
            CachedKind::BindGroupLayout,
            key,
            || Ok(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })),
        )
    }

    /// The pipeline layout made of the cached bind-group layouts `groups` (group 0 first; `None` for
    /// an unused group) under `key`, created at first use. A group key with no cached layout is
    /// [`GpuError::Stale`]; more than [`MAX_BIND_GROUPS`] groups is [`GpuError::Invalid`].
    pub fn pipeline_layout(
        &mut self,
        key: K,
        label: &str,
        groups: &[Option<&K>],
    ) -> Result<&wgpu::PipelineLayout, GpuError> {
        if groups.len() > MAX_BIND_GROUPS {
            return Err(GpuError::Invalid { what: "a pipeline layout with more than 8 bind groups" });
        }
        let device = &self.device;
        let layouts = &self.bind_group_layouts;
        get_or_create(
            &mut self.pipeline_layouts,
            &mut self.stats,
            self.warm,
            device,
            CachedKind::PipelineLayout,
            key,
            || {
                let mut resolved: [Option<&wgpu::BindGroupLayout>; MAX_BIND_GROUPS] = [None; MAX_BIND_GROUPS];
                for (slot, group) in resolved.iter_mut().zip(groups) {
                    if let Some(group) = group {
                        *slot = Some(layouts.get(group).ok_or(GpuError::Stale { what: "bind-group layout key" })?);
                    }
                }
                Ok(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &resolved[..groups.len()],
                    immediate_size: 0,
                }))
            },
        )
    }

    /// The render pipeline under `key`; at first use `create` builds it from the cached modules and
    /// layouts in [`Parts`].
    pub fn render_pipeline(
        &mut self,
        key: K,
        create: impl FnOnce(&Parts<'_, K>) -> Result<wgpu::RenderPipeline, GpuError>,
    ) -> Result<&wgpu::RenderPipeline, GpuError> {
        let parts = Parts {
            device: &self.device,
            shaders: &self.shaders,
            bind_group_layouts: &self.bind_group_layouts,
            pipeline_layouts: &self.pipeline_layouts,
        };
        get_or_create(
            &mut self.render,
            &mut self.stats,
            self.warm,
            &self.device,
            CachedKind::RenderPipeline,
            key,
            || create(&parts),
        )
    }

    /// The compute pipeline under `key`; at first use `create` builds it from [`Parts`].
    pub fn compute_pipeline(
        &mut self,
        key: K,
        create: impl FnOnce(&Parts<'_, K>) -> Result<wgpu::ComputePipeline, GpuError>,
    ) -> Result<&wgpu::ComputePipeline, GpuError> {
        let parts = Parts {
            device: &self.device,
            shaders: &self.shaders,
            bind_group_layouts: &self.bind_group_layouts,
            pipeline_layouts: &self.pipeline_layouts,
        };
        get_or_create(
            &mut self.compute,
            &mut self.stats,
            self.warm,
            &self.device,
            CachedKind::ComputePipeline,
            key,
            || create(&parts),
        )
    }

    /// The bind group under `key`, made at first use from the cached layout `layout` and `entries`.
    /// When a bound resource is replaced (a resize, a rebuild), call
    /// [`PipelineCache::invalidate_bind_group`] so the next call makes a new one.
    pub fn bind_group(
        &mut self,
        key: K,
        label: &str,
        layout: &K,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> Result<&wgpu::BindGroup, GpuError> {
        let device = &self.device;
        let layouts = &self.bind_group_layouts;
        get_or_create(&mut self.bind_groups, &mut self.stats, self.warm, device, CachedKind::BindGroup, key, || {
            let layout = layouts.get(layout).ok_or(GpuError::Stale { what: "bind-group layout key" })?;
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries }))
        })
    }

    /// Forgets a bind group, so the next [`PipelineCache::bind_group`] makes it again. Returns
    /// whether there was one.
    pub fn invalidate_bind_group(&mut self, key: &K) -> bool {
        self.bind_groups.remove(key).is_some()
    }

    /// A cached shader module. Never allocates.
    pub fn get_shader(&self, key: &K) -> Option<&wgpu::ShaderModule> {
        self.shaders.get(key)
    }

    /// A cached bind-group layout. Never allocates.
    pub fn get_bind_group_layout(&self, key: &K) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layouts.get(key)
    }

    /// A cached pipeline layout. Never allocates.
    pub fn get_pipeline_layout(&self, key: &K) -> Option<&wgpu::PipelineLayout> {
        self.pipeline_layouts.get(key)
    }

    /// A cached render pipeline. Never allocates: the per-frame lookup.
    pub fn get_render_pipeline(&self, key: &K) -> Option<&wgpu::RenderPipeline> {
        self.render.get(key)
    }

    /// A cached compute pipeline. Never allocates: the per-frame lookup.
    pub fn get_compute_pipeline(&self, key: &K) -> Option<&wgpu::ComputePipeline> {
        self.compute.get(key)
    }

    /// A cached bind group. Never allocates: the per-frame lookup.
    pub fn get_bind_group(&self, key: &K) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(key)
    }

    /// Number of cached objects of one kind.
    pub fn len(&self, kind: CachedKind) -> usize {
        match kind {
            CachedKind::Shader => self.shaders.len(),
            CachedKind::BindGroupLayout => self.bind_group_layouts.len(),
            CachedKind::PipelineLayout => self.pipeline_layouts.len(),
            CachedKind::RenderPipeline => self.render.len(),
            CachedKind::ComputePipeline => self.compute.len(),
            CachedKind::BindGroup => self.bind_groups.len(),
        }
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        CachedKind::ALL.into_iter().all(|kind| self.len(kind) == 0)
    }

    /// Empties the cache and moves it to `gpu`'s device, after a device loss `[ERR-05]`. The cache
    /// is cold again: the owner re-runs its pre-warm and calls [`PipelineCache::mark_warm`].
    pub fn reset(&mut self, gpu: &Gpu) {
        self.device = gpu.device.clone();
        self.shaders.clear();
        self.bind_group_layouts.clear();
        self.pipeline_layouts.clear();
        self.render.clear();
        self.compute.clear();
        self.bind_groups.clear();
        self.warm = false;
        self.stats.resets += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    /// Test-only WGSL: doubles every element of a storage buffer.
    const DOUBLE: &str = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) { data[id.x] = data[id.x] * 2u; }
}";

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Key {
        Double,
        Broken,
        Data,
        Missing,
    }

    const STORAGE_ENTRY: wgpu::BindGroupLayoutEntry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    /// Builds the doubling pipeline the way an owner's pre-warm would.
    fn warm_up(cache: &mut PipelineCache<Key>) -> Result<(), GpuError> {
        cache.shader(Key::Double, "double", DOUBLE)?;
        cache.bind_group_layout(Key::Double, "double", &[STORAGE_ENTRY])?;
        cache.pipeline_layout(Key::Double, "double", &[Some(&Key::Double)])?;
        cache.compute_pipeline(Key::Double, |parts| {
            Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("double"),
                layout: Some(parts.pipeline_layout(&Key::Double)?),
                module: parts.shader(&Key::Double)?,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        })?;
        Ok(())
    }

    #[test]
    fn perf_r4_objects_are_created_once_and_looked_up_after() {
        let gpu = testing::gpu();
        let mut cache = PipelineCache::new(gpu);
        assert!(cache.is_empty());
        warm_up(&mut cache).unwrap();
        cache.mark_warm();
        for _ in 0..50 {
            warm_up(&mut cache).unwrap();
        }
        let stats = *cache.stats();
        assert_eq!(stats.total_created(), 4, "{stats:?}");
        assert_eq!(stats.hits, 50 * 4);
        assert_eq!(stats.late, 0, "nothing created after the pre-warm");
        for kind in [CachedKind::Shader, CachedKind::BindGroupLayout, CachedKind::PipelineLayout] {
            assert_eq!((stats.of(kind), cache.len(kind)), (1, 1), "{}", kind.name());
        }
        assert!(cache.get_compute_pipeline(&Key::Double).is_some());
        assert!(cache.get_render_pipeline(&Key::Double).is_none(), "kinds have separate maps");

        // A late creation is counted, a late bind group is not.
        cache.shader(Key::Data, "late", DOUBLE).unwrap();
        assert_eq!(cache.stats().late, 1);

        // The cached pipeline really runs: 0..64 doubled.
        let values: Vec<u32> = (0..64).collect();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("data"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, crate::pod::slice_as_bytes(&values));
        let entries = [wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }];
        cache.bind_group(Key::Data, "data", &Key::Double, &entries).unwrap();
        cache.bind_group(Key::Data, "data", &Key::Double, &entries).unwrap();
        assert_eq!(cache.stats().of(CachedKind::BindGroup), 1);
        assert_eq!(cache.stats().late, 1, "bind groups are not late creations");
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(cache.get_compute_pipeline(&Key::Double).unwrap());
            pass.set_bind_group(0, cache.get_bind_group(&Key::Data).unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let bytes = testing::read_buffer(gpu, &buffer, 0, 256);
        let mut doubled = [0_u32; 64];
        crate::pod::copy_from_bytes(&bytes, &mut doubled).unwrap();
        assert!(doubled.iter().enumerate().all(|(i, &v)| v == 2 * i as u32), "{doubled:?}");

        assert!(cache.invalidate_bind_group(&Key::Data));
        assert!(!cache.invalidate_bind_group(&Key::Data));
        assert!(cache.get_bind_group(&Key::Data).is_none());
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn err_02_bad_shaders_and_missing_parts_are_errors_and_not_cached() {
        let gpu = testing::gpu();
        let mut cache = PipelineCache::new(gpu);
        let error = cache.shader(Key::Broken, "broken", "fn main( {").unwrap_err();
        assert!(matches!(error, GpuError::Validation { scope: "shader module", .. }), "{error}");
        assert!(cache.get_shader(&Key::Broken).is_none());
        assert_eq!(cache.stats().failed, 1);

        // A layout naming a group that was never made, and a pipeline built from missing parts.
        let missing = cache.pipeline_layout(Key::Missing, "missing", &[Some(&Key::Missing)]);
        assert!(matches!(missing, Err(GpuError::Stale { .. })));
        let pipeline = cache.compute_pipeline(Key::Missing, |parts| {
            Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("missing"),
                layout: None,
                module: parts.shader(&Key::Missing)?,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        });
        assert!(matches!(pipeline, Err(GpuError::Stale { what: "shader module key" })));
        let nine = [None; 9];
        assert!(matches!(cache.pipeline_layout(Key::Missing, "nine", &nine), Err(GpuError::Invalid { .. })));
        let group = cache.bind_group(Key::Missing, "missing", &Key::Missing, &[]);
        assert!(matches!(group, Err(GpuError::Stale { .. })));
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_created(), 0);
        assert_eq!(gpu.events().pending(), 0, "the errors were captured, not left uncaptured");
    }

    #[test]
    fn err_05_reset_empties_the_cache_for_a_new_device() {
        let gpu = testing::gpu();
        let mut cache = PipelineCache::new(gpu);
        warm_up(&mut cache).unwrap();
        cache.mark_warm();
        assert!(cache.is_warm());
        cache.reset(gpu);
        assert!(cache.is_empty() && !cache.is_warm());
        assert_eq!(cache.stats().resets, 1);
        warm_up(&mut cache).unwrap();
        assert_eq!(cache.stats().total_created(), 8, "made again after the reset");
        assert_eq!(cache.stats().late, 0);
        assert!(format!("{cache:?}").contains("compute: 1"));
    }
}
