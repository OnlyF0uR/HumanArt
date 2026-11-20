//! The frame graph `[RND-01]`.
//!
//! Passes are declared in execution order with the resources they read and write. [`GraphBuilder::compile`]
//! then
//!
//! - records for every read which earlier pass produced the data (a read of a transient resource
//!   that nothing wrote is an error),
//! - **culls** every pass whose writes reach neither an output nor a pass that is kept (passes
//!   marked [`PassBuilder::side_effect`] are always kept),
//! - computes each transient resource's **lifetime** (first and last use among the kept passes),
//! - **aliases** transient textures: resources with the same size, layers and format whose lifetimes
//!   do not overlap share one physical texture (greedy interval colouring in order of first use),
//! - marks the first write of every physical texture in the frame, so the pass clears instead of
//!   loading.
//!
//! wgpu inserts the barriers itself; the graph decides ordering, allocation and culling. Compiling
//! allocates (it runs when the configuration changes: a resize, a debug view, a quality tier);
//! walking a [`CompiledGraph`] each frame does not.
//!
//! The pass payload `P` is the caller's own (an enum of its pass kinds), so executing a frame is a
//! `match` over plain data, with no trait objects `[PERF-R1]`.

use crate::error::GraphError;

/// The most passes one graph may declare.
pub const MAX_PASSES: usize = 64;
/// The most resources one graph may declare.
pub const MAX_RESOURCES: usize = 128;
/// The most reads plus writes of one pass.
pub const MAX_ACCESSES: usize = 16;

/// Names a resource of a [`GraphBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u16);

impl ResourceId {
    /// The index in declaration order.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Names a pass of a [`GraphBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(u16);

impl PassId {
    /// The index in declaration order.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What a texture resource is: size, layers, format and how passes use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureInfo {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Array layers.
    pub layers: u32,
    /// Texel format.
    pub format: wgpu::TextureFormat,
    /// Usages; aliased resources get the union.
    pub usage: wgpu::TextureUsages,
}

impl TextureInfo {
    /// A one-layer 2D texture.
    pub const fn d2(width: u32, height: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        Self { width, height, layers: 1, format, usage }
    }

    /// Whether two resources may share one physical texture.
    fn aliases_with(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.layers == other.layers
            && self.format == other.format
    }
}

/// Whether a resource lives only inside the frame or is owned outside the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Created by the graph, may be aliased, content undefined at the start of the frame.
    Transient,
    /// Owned by the caller (history buffers, shadow atlases, the output): never aliased, its
    /// content at the start of the frame may be read.
    Imported,
}

#[derive(Debug, Clone)]
struct ResourceDecl {
    name: &'static str,
    info: TextureInfo,
    lifetime: Lifetime,
    output: bool,
}

#[derive(Debug, Clone)]
struct PassDecl<P> {
    name: &'static str,
    payload: P,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
    side_effect: bool,
}

/// Declares passes and resources; see the module documentation.
#[derive(Debug, Clone)]
pub struct GraphBuilder<P> {
    resources: Vec<ResourceDecl>,
    passes: Vec<PassDecl<P>>,
    error: Option<GraphError>,
}

impl<P: Copy> Default for GraphBuilder<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Copy> GraphBuilder<P> {
    /// An empty graph.
    pub fn new() -> Self {
        Self { resources: Vec::new(), passes: Vec::new(), error: None }
    }

    fn add_resource(&mut self, name: &'static str, info: TextureInfo, lifetime: Lifetime) -> ResourceId {
        if self.resources.len() >= MAX_RESOURCES {
            self.error.get_or_insert(GraphError::TooMany { what: "resources", limit: MAX_RESOURCES });
            return ResourceId(u16::MAX);
        }
        if info.width == 0 || info.height == 0 || info.layers == 0 {
            self.error.get_or_insert(GraphError::EmptyResource { resource: name });
        }
        self.resources.push(ResourceDecl { name, info, lifetime, output: false });
        ResourceId((self.resources.len() - 1) as u16)
    }

    /// A transient texture: created and possibly aliased by the graph.
    pub fn texture(&mut self, name: &'static str, info: TextureInfo) -> ResourceId {
        self.add_resource(name, info, Lifetime::Transient)
    }

    /// A texture owned outside the graph (a history buffer, a shadow atlas, the display target).
    pub fn import(&mut self, name: &'static str, info: TextureInfo) -> ResourceId {
        self.add_resource(name, info, Lifetime::Imported)
    }

    /// Marks `resource` as a result of the frame: the passes producing it are kept.
    pub fn output(&mut self, resource: ResourceId) {
        match self.resources.get_mut(resource.index()) {
            Some(r) => r.output = true,
            None => {
                self.error.get_or_insert(GraphError::UnknownResource { pass: "<output>" });
            }
        }
    }

    /// Declares the next pass in execution order.
    pub fn pass(&mut self, name: &'static str, payload: P) -> PassBuilder<'_, P> {
        if self.passes.len() >= MAX_PASSES {
            self.error.get_or_insert(GraphError::TooMany { what: "passes", limit: MAX_PASSES });
            return PassBuilder { graph: self, index: None };
        }
        self.passes.push(PassDecl { name, payload, reads: Vec::new(), writes: Vec::new(), side_effect: false });
        let index = Some(self.passes.len() - 1);
        PassBuilder { graph: self, index }
    }

    /// Resolves dependencies, culls, computes lifetimes and aliases; see the module documentation.
    pub fn compile(self) -> Result<CompiledGraph<P>, GraphError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let pass_count = self.passes.len();
        let resource_count = self.resources.len();
        // 1. Producers: for every read, the pass that last wrote the resource before it.
        let mut last_writer: Vec<Option<usize>> = vec![None; resource_count];
        let mut depends: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
        for (p, pass) in self.passes.iter().enumerate() {
            for r in pass.reads.iter().chain(&pass.writes) {
                if r.index() >= resource_count {
                    return Err(GraphError::UnknownResource { pass: pass.name });
                }
            }
            for &r in &pass.reads {
                match last_writer[r.index()] {
                    Some(w) => depends[p].push(w),
                    None if self.resources[r.index()].lifetime == Lifetime::Transient => {
                        return Err(GraphError::ReadBeforeWrite {
                            pass: pass.name,
                            resource: self.resources[r.index()].name,
                        });
                    }
                    None => {}
                }
            }
            for &r in &pass.writes {
                // A partial write keeps what an earlier writer left: order after it.
                if let Some(w) = last_writer[r.index()] {
                    depends[p].push(w);
                }
                last_writer[r.index()] = Some(p);
            }
        }
        // 2. Culling: keep side effects and the final writers of outputs, then everything they need.
        let mut live = vec![false; pass_count];
        let mut stack: Vec<usize> = Vec::new();
        for (p, pass) in self.passes.iter().enumerate() {
            if pass.side_effect {
                stack.push(p);
            }
        }
        for (r, res) in self.resources.iter().enumerate() {
            if res.output {
                match last_writer[r] {
                    Some(w) => stack.push(w),
                    None => return Err(GraphError::OutputNeverWritten { resource: res.name }),
                }
            }
        }
        while let Some(p) = stack.pop() {
            if live[p] {
                continue;
            }
            live[p] = true;
            stack.extend(depends[p].iter().copied().filter(|&d| !live[d]));
        }
        let order: Vec<usize> = (0..pass_count).filter(|&p| live[p]).collect();
        // 3. Lifetimes over the kept passes (positions in `order`).
        let mut first_use: Vec<Option<usize>> = vec![None; resource_count];
        let mut last_use: Vec<usize> = vec![0; resource_count];
        for (position, &p) in order.iter().enumerate() {
            let pass = &self.passes[p];
            for r in pass.reads.iter().chain(&pass.writes) {
                first_use[r.index()].get_or_insert(position);
                last_use[r.index()] = position;
            }
        }
        // 4. Aliasing: greedy interval colouring of the transient resources by first use.
        let mut transient: Vec<usize> = (0..resource_count)
            .filter(|&r| self.resources[r].lifetime == Lifetime::Transient && first_use[r].is_some())
            .collect();
        transient.sort_by_key(|&r| (first_use[r], r));
        let mut slots: Vec<PhysicalTexture> = Vec::new();
        let mut slot_free_after: Vec<usize> = Vec::new();
        let mut physical: Vec<Option<u16>> = vec![None; resource_count];
        for &r in &transient {
            let decl = &self.resources[r];
            let start = first_use[r].unwrap_or(0);
            let reuse = slots
                .iter()
                .zip(&slot_free_after)
                .position(|(slot, &free_after)| free_after < start && slot.info.aliases_with(&decl.info));
            let slot = if let Some(s) = reuse {
                slots[s].info.usage |= decl.info.usage;
                slots[s].names.push(decl.name);
                s
            } else {
                slots.push(PhysicalTexture { info: decl.info, names: vec![decl.name] });
                slot_free_after.push(0);
                slots.len() - 1
            };
            slot_free_after[slot] = last_use[r];
            physical[r] = Some(slot as u16);
        }
        // 5. Compiled passes with first-write flags per physical texture (or imported resource).
        let mut written: Vec<bool> = vec![false; slots.len()];
        let mut passes = Vec::with_capacity(order.len());
        for &p in &order {
            let pass = &self.passes[p];
            let mut writes = Vec::with_capacity(pass.writes.len());
            for &r in &pass.writes {
                let first = match physical[r.index()] {
                    Some(s) => !std::mem::replace(&mut written[s as usize], true),
                    None => false,
                };
                // A read-modify-write of a transient resource never clears.
                let first = first && !pass.reads.contains(&r);
                writes.push(Write { resource: r, first });
            }
            passes.push(CompiledPass {
                id: PassId(p as u16),
                name: pass.name,
                payload: pass.payload,
                reads: pass.reads.clone(),
                writes,
            });
        }
        let culled = (0..pass_count).filter(|&p| !live[p]).map(|p| self.passes[p].name).collect();
        let lifetimes = (0..resource_count).map(|r| first_use[r].map(|f| (f, last_use[r]))).collect();
        Ok(CompiledGraph {
            names: self.resources.iter().map(|r| r.name).collect(),
            infos: self.resources.iter().map(|r| r.info).collect(),
            lifetimes,
            physical,
            slots,
            passes,
            culled,
        })
    }
}

/// Adds accesses to a pass; returned by [`GraphBuilder::pass`].
#[derive(Debug)]
pub struct PassBuilder<'a, P> {
    graph: &'a mut GraphBuilder<P>,
    index: Option<usize>,
}

impl<P> PassBuilder<'_, P> {
    fn access(&mut self, resource: ResourceId, write: bool) {
        let Some(index) = self.index else { return };
        let Some(pass) = self.graph.passes.get_mut(index) else { return };
        if pass.reads.len() + pass.writes.len() >= MAX_ACCESSES {
            self.graph.error.get_or_insert(GraphError::TooMany { what: "accesses in one pass", limit: MAX_ACCESSES });
            return;
        }
        if write { pass.writes.push(resource) } else { pass.reads.push(resource) }
    }

    /// The pass samples or loads `resource`.
    pub fn read(mut self, resource: ResourceId) -> Self {
        self.access(resource, false);
        self
    }

    /// The pass renders or stores into `resource`.
    pub fn write(mut self, resource: ResourceId) -> Self {
        self.access(resource, true);
        self
    }

    /// The pass has effects outside the graph (a readback, a timestamp, a picking copy): never culled.
    pub fn side_effect(self) -> Self {
        if let Some(pass) = self.index.and_then(|i| self.graph.passes.get_mut(i)) {
            pass.side_effect = true;
        }
        self
    }

    /// The id of the pass (or `None` past [`MAX_PASSES`], which `compile` reports).
    pub fn id(&self) -> Option<PassId> {
        self.index.map(|i| PassId(i as u16))
    }
}

/// One physical texture after aliasing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalTexture {
    /// The texture to create (the usage is the union over the aliased resources).
    pub info: TextureInfo,
    /// The resources that share it, in order of first use.
    pub names: Vec<&'static str>,
}

/// A write of a compiled pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Write {
    /// The resource.
    pub resource: ResourceId,
    /// The first write of its physical texture in the frame: clear rather than load.
    pub first: bool,
}

/// A pass that survived culling.
#[derive(Debug, Clone)]
pub struct CompiledPass<P> {
    /// Its id in the builder.
    pub id: PassId,
    /// Its name (the timestamp scope).
    pub name: &'static str,
    /// The caller's payload.
    pub payload: P,
    /// What it reads.
    pub reads: Vec<ResourceId>,
    /// What it writes.
    pub writes: Vec<Write>,
}

impl<P> CompiledPass<P> {
    /// Whether the pass's write of `resource` is the first in the frame (clear), `false` when it
    /// loads, or when the pass does not write it.
    pub fn clears(&self, resource: ResourceId) -> bool {
        self.writes.iter().any(|w| w.resource == resource && w.first)
    }
}

/// The result of [`GraphBuilder::compile`]. Walking it allocates nothing.
#[derive(Debug, Clone)]
pub struct CompiledGraph<P> {
    names: Vec<&'static str>,
    infos: Vec<TextureInfo>,
    lifetimes: Vec<Option<(usize, usize)>>,
    physical: Vec<Option<u16>>,
    slots: Vec<PhysicalTexture>,
    passes: Vec<CompiledPass<P>>,
    culled: Vec<&'static str>,
}

impl<P> CompiledGraph<P> {
    /// The kept passes in execution order.
    pub fn passes(&self) -> &[CompiledPass<P>] {
        &self.passes
    }

    /// The names of the culled passes.
    pub fn culled(&self) -> &[&'static str] {
        &self.culled
    }

    /// Whether the pass named `name` is kept.
    pub fn is_live(&self, name: &str) -> bool {
        self.passes.iter().any(|p| p.name == name)
    }

    /// The physical textures to create for the transient resources.
    pub fn physical_textures(&self) -> &[PhysicalTexture] {
        &self.slots
    }

    /// The physical texture of a transient resource; `None` for imported or unused ones.
    pub fn physical(&self, resource: ResourceId) -> Option<usize> {
        self.physical.get(resource.index()).copied().flatten().map(usize::from)
    }

    /// The first and last position (in [`Self::passes`]) at which a resource is used.
    pub fn lifetime(&self, resource: ResourceId) -> Option<(usize, usize)> {
        self.lifetimes.get(resource.index()).copied().flatten()
    }

    /// A resource's declaration.
    pub fn info(&self, resource: ResourceId) -> Option<&TextureInfo> {
        self.infos.get(resource.index())
    }

    /// A resource's name.
    pub fn name(&self, resource: ResourceId) -> Option<&'static str> {
        self.names.get(resource.index()).copied()
    }

    /// Texture bytes of the physical transient textures (one mip level), and of the transient
    /// resources had nothing been aliased.
    pub fn transient_bytes(&self) -> (u64, u64) {
        let bytes =
            |i: &TextureInfo| humanart_gpu::resources::level_bytes(i.format, i.width, i.height) * u64::from(i.layers);
        let physical = self.slots.iter().map(|s| bytes(&s.info)).sum();
        let unaliased = (0..self.names.len())
            .filter(|&r| self.physical.get(r).copied().flatten().is_some())
            .filter_map(|r| self.infos.get(r))
            .map(bytes)
            .sum();
        (physical, unaliased)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const RT: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT;
    const TEX: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;

    fn tex(w: u32) -> TextureInfo {
        TextureInfo::d2(w, w, F16, RT.union(TEX))
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        A,
        B,
        C,
        D,
    }

    #[test]
    fn rnd_01_culls_passes_whose_results_are_unused() {
        let mut g = GraphBuilder::new();
        let a = g.texture("a", tex(4));
        let unused = g.texture("unused", tex(4));
        let out = g.import("out", tex(4));
        g.pass("write a", Kind::A).write(a);
        g.pass("debug", Kind::B).read(a).write(unused);
        g.pass("final", Kind::C).read(a).write(out);
        g.output(out);
        let c = g.compile().unwrap();
        let names: Vec<_> = c.passes().iter().map(|p| p.name).collect();
        assert_eq!(names, ["write a", "final"]);
        assert_eq!(c.culled(), ["debug"]);
        assert!(c.physical(unused).is_none());
        assert!(c.physical(out).is_none(), "imported resources are never aliased");
    }

    #[test]
    fn rnd_01_side_effect_passes_are_kept_with_their_inputs() {
        let mut g = GraphBuilder::new();
        let ids = g.texture("ids", TextureInfo::d2(8, 8, wgpu::TextureFormat::R32Uint, RT));
        g.pass("pick", Kind::A).write(ids);
        g.pass("readback", Kind::B).read(ids).side_effect();
        g.pass("orphan", Kind::C);
        let c = g.compile().unwrap();
        assert!(c.is_live("pick") && c.is_live("readback"));
        assert!(!c.is_live("orphan"));
    }

    #[test]
    fn rnd_01_aliases_textures_with_disjoint_lifetimes() {
        let mut g = GraphBuilder::new();
        let a = g.texture("a", tex(16));
        let b = g.texture("b", tex(16));
        let c_ = g.texture("c", tex(16));
        let small = g.texture("small", tex(8));
        let out = g.import("out", tex(16));
        g.pass("1", Kind::A).write(a);
        g.pass("2", Kind::B).read(a).write(b);
        g.pass("3", Kind::C).read(b).write(c_).write(small);
        g.pass("4", Kind::D).read(c_).read(small).write(out);
        g.output(out);
        let compiled = g.compile().unwrap();
        // a lives over passes 0..=1, b 1..=2, c 2..=3: a and c may share, b may not share with either.
        assert_eq!(compiled.lifetime(a), Some((0, 1)));
        assert_eq!(compiled.lifetime(c_), Some((2, 3)));
        assert_eq!(compiled.physical(a), compiled.physical(c_));
        assert_ne!(compiled.physical(a), compiled.physical(b));
        assert_ne!(compiled.physical(small), compiled.physical(a), "different sizes never alias");
        assert_eq!(compiled.physical_textures().len(), 3);
        let (physical, unaliased) = compiled.transient_bytes();
        assert!(physical < unaliased);
        assert_eq!(compiled.physical_textures()[compiled.physical(a).unwrap()].names, ["a", "c"]);
    }

    #[test]
    fn rnd_01_first_write_clears_and_later_writes_load() {
        let mut g = GraphBuilder::new();
        let hdr = g.texture("hdr", tex(4));
        let out = g.import("out", tex(4));
        g.pass("opaque", Kind::A).write(hdr);
        g.pass("overlay", Kind::B).write(hdr);
        g.pass("resolve", Kind::C).read(hdr).write(out);
        g.output(out);
        let c = g.compile().unwrap();
        assert!(c.passes()[0].clears(hdr));
        assert!(!c.passes()[1].clears(hdr));
        assert!(!c.passes()[2].clears(out), "imported targets are loaded, the pass decides");
    }

    #[test]
    fn rnd_01_rejects_bad_graphs() {
        let mut g: GraphBuilder<Kind> = GraphBuilder::new();
        let a = g.texture("a", tex(4));
        g.pass("reader", Kind::A).read(a).side_effect();
        assert_eq!(g.compile().unwrap_err(), GraphError::ReadBeforeWrite { pass: "reader", resource: "a" });

        let mut g: GraphBuilder<Kind> = GraphBuilder::new();
        let out = g.import("out", tex(4));
        g.output(out);
        assert_eq!(g.compile().unwrap_err(), GraphError::OutputNeverWritten { resource: "out" });

        let mut g: GraphBuilder<Kind> = GraphBuilder::new();
        g.texture("empty", tex(0));
        assert_eq!(g.compile().unwrap_err(), GraphError::EmptyResource { resource: "empty" });

        let mut g: GraphBuilder<Kind> = GraphBuilder::new();
        for _ in 0..=MAX_PASSES {
            g.pass("p", Kind::A).side_effect();
        }
        assert!(matches!(g.compile().unwrap_err(), GraphError::TooMany { what: "passes", .. }));

        let mut other: GraphBuilder<Kind> = GraphBuilder::new();
        let foreign = other.texture("x", tex(4));
        let mut g: GraphBuilder<Kind> = GraphBuilder::new();
        g.pass("p", Kind::A).write(foreign).side_effect();
        assert!(matches!(g.compile().unwrap_err(), GraphError::UnknownResource { .. }));
    }

    #[test]
    fn rnd_01_imported_history_may_be_read_before_written() {
        let mut g = GraphBuilder::new();
        let history = g.import("history", tex(4));
        let color = g.texture("color", tex(4));
        let out = g.import("out", tex(4));
        g.pass("scene", Kind::A).write(color);
        g.pass("taa", Kind::B).read(color).read(history).write(out);
        g.pass("copy history", Kind::C).read(out).write(history);
        g.output(out);
        g.output(history);
        let c = g.compile().unwrap();
        assert_eq!(c.passes().len(), 3);
    }

    /// Random graphs: the compiled order respects every dependency and aliased resources never
    /// overlap in time.
    #[test]
    fn rnd_01_random_graphs_keep_order_and_never_overlap_aliases() {
        let mut rng = humanart_core::rng::Rng::new(0x5EED_0011);
        for _ in 0..200 {
            let mut g = GraphBuilder::new();
            let n_res = 2 + rng.index(10);
            let res: Vec<ResourceId> = (0..n_res).map(|_| g.texture("r", tex(4 + 4 * rng.below(2) as u32))).collect();
            let out = g.import("out", tex(4));
            let mut written = vec![false; n_res];
            let n_pass = 1 + rng.index(12);
            for p in 0..n_pass {
                let w = rng.index(n_res);
                let mut b = g.pass("p", p);
                let r = rng.index(n_res);
                if written[r] {
                    b = b.read(res[r]);
                }
                b = b.write(res[w]);
                if p == n_pass - 1 {
                    b = b.write(out);
                }
                let _ = b;
                written[w] = true;
            }
            g.output(out);
            let c = g.compile().unwrap();
            let positions: Vec<usize> = c.passes().iter().map(|p| p.payload).collect();
            assert!(positions.windows(2).all(|w| w[0] < w[1]), "declaration order is kept");
            for (i, &a) in res.iter().enumerate() {
                for &b in &res[i + 1..] {
                    if let (Some(pa), Some(pb)) = (c.physical(a), c.physical(b))
                        && pa == pb
                    {
                        let (fa, la) = c.lifetime(a).unwrap();
                        let (fb, lb) = c.lifetime(b).unwrap();
                        assert!(la < fb || lb < fa, "aliased resources overlap");
                    }
                }
            }
        }
    }
}
