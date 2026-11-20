//! Hair-card LODs `[HAIR-10]` `[IO-12]`: the card maps of every card of a
//! [`CardSet`](humanart_model::hair::cards::CardSet) baked into one atlas with the `[BAKE-06]`
//! card baker, and the card LODs written as one glTF binary.
//!
//! **Baking.** Each card's member strands are moved into the card's straightened space
//! ([`HairCard::straighten`]) and baked on the flat card `origin = 0`, `across = (width, 0, 0)`,
//! `along = (0, length, 0)` (normal `−Z`, so the depth map's 1 is `−z`), `depth = card.depth`, at
//! `cell` texels; the card's maps are copied into its atlas cell. Strand ids are the groom's strand
//! indices, so the id map matches the strand groom's colour variation.
//!
//! **glTF.** One mesh per LOD (`hair_cards_lod0`, ...), one node each, all in the default scene;
//! one material per LOD, double-sided, `alphaMode` `BLEND` (the baked alpha is the physical
//! fractional coverage of the strands, not a mask), whose base colour texture is white with the
//! coverage as alpha. The other maps are embedded as PNG images and
//! textures named `hair_lod<k>_<map>` (`depth`, `root`, `id`, `flow`; flow is two channels, the
//! direction mapped from `[−1, 1]` to `[0, 1]`) for engines to bind to their own hair card shader.

use humanart_io::gltf::{
    AlphaMode, Builder, ElementType, Material, Mesh, Node, Pbr, Primitive, Scene, Target, Texture, TextureInfo,
};
use humanart_io::{ColorSpace, Format, Image, Samples, png};
use humanart_math::Vec3;
use humanart_model::hair::cards::{CardSet, HairCard};

use super::{GroomToolError, Result};
use crate::bake::hair_card::{Card, HairCardMaps, Strands as BakeStrands, bake_hair_card};
use crate::bake::{BakeMap, BakeTarget};

/// The largest atlas side, texels.
pub const MAX_ATLAS_SIDE: u32 = 8192;

/// The strands a card set was fitted to: `per` world points each and the root and tip widths.
#[derive(Clone, Copy, Debug)]
pub struct CardStrands<'a> {
    /// World points, `per` per strand, root first.
    pub world: &'a [Vec3],
    /// Points per strand.
    pub per: u32,
    /// Root diameter per strand, metres.
    pub width_root: &'a [f32],
    /// Tip diameter per strand, metres.
    pub width_tip: &'a [f32],
}

/// The baked atlas of a card set: every map `cols · cell[0]` by `rows · cell[1]` texels.
#[derive(Clone, Debug, PartialEq)]
pub struct CardAtlas {
    /// Texels per card cell, (width, height).
    pub cell: [u32; 2],
    /// Coverage.
    pub alpha: BakeMap,
    /// Normalised depth.
    pub depth: BakeMap,
    /// Root-to-tip parameter.
    pub root: BakeMap,
    /// Per-strand random id.
    pub id: BakeMap,
    /// Root-to-tip direction in texture space (2 channels).
    pub flow: BakeMap,
}

/// The member strands of `card` in its straightened space.
fn straightened(card: &HairCard, strands: &CardStrands<'_>) -> Result<BakeStrands> {
    let per = strands.per as usize;
    let n = strands.width_root.len();
    let mut out = BakeStrands::default();
    out.offsets.push(0);
    for &i in &card.members {
        let i = i as usize;
        let (Some(pts), Some(&wr), Some(&wt)) =
            (strands.world.get(i * per..(i + 1) * per), strands.width_root.get(i), strands.width_tip.get(i))
        else {
            return Err(GroomToolError::Invalid("card member strand out of range"));
        };
        if i >= n {
            return Err(GroomToolError::Invalid("card member strand out of range"));
        }
        out.points.extend(pts.iter().map(|&p| card.straighten(p)));
        out.offsets.push(out.points.len() as u32);
        out.root_width.push(wr);
        out.tip_width.push(wt);
        out.ids.push(i as u32);
    }
    Ok(out)
}

/// Copies `src` (one cell) into `dst` at texel `(x0, y0)`.
fn blit(dst: &mut BakeMap, src: &BakeMap, x0: u32, y0: u32) {
    let c = src.channels as usize;
    for y in 0..src.height {
        let s = (y * src.width) as usize;
        let d = ((y0 + y) * dst.width + x0) as usize;
        let w = src.width as usize;
        if let (Some(from), Some(to)) = (src.data.get(s * c..(s + w) * c), dst.data.get_mut(d * c..(d + w) * c)) {
            to.copy_from_slice(from);
        }
        if let (Some(from), Some(to)) = (src.coverage.get(s..s + w), dst.coverage.get_mut(d..d + w)) {
            to.copy_from_slice(from);
        }
    }
}

/// Bakes the maps of every card of `set` into an atlas of `cell`-sized cells.
pub fn bake_card_atlas(set: &CardSet, strands: &CardStrands<'_>, cell: [u32; 2]) -> Result<CardAtlas> {
    let n = strands.width_root.len();
    if strands.width_tip.len() != n || strands.world.len() != n * strands.per as usize || strands.per < 2 {
        return Err(GroomToolError::Invalid("card strands: arrays differ in length"));
    }
    if set.cols == 0 || set.rows == 0 || (set.cols * set.rows) < set.cards.len() as u32 {
        return Err(GroomToolError::Invalid("card set: the atlas grid does not hold every card"));
    }
    let (w, h) = (u64::from(set.cols) * u64::from(cell[0]), u64::from(set.rows) * u64::from(cell[1]));
    if cell[0] == 0 || cell[1] == 0 || w > u64::from(MAX_ATLAS_SIDE) || h > u64::from(MAX_ATLAS_SIDE) {
        return Err(GroomToolError::Invalid("card atlas: cell size zero or atlas larger than 8192 texels"));
    }
    let target = BakeTarget::new(0, w as u32, h as u32);
    let mut atlas = CardAtlas {
        cell,
        alpha: BakeMap::new(&target, 1, 0.0)?,
        depth: BakeMap::new(&target, 1, 0.5)?,
        root: BakeMap::new(&target, 1, 0.0)?,
        id: BakeMap::new(&target, 1, 0.0)?,
        flow: BakeMap::new(&target, 2, 0.0)?,
    };
    for (index, card) in set.cards.iter().enumerate() {
        let index = index as u32;
        let flat = Card {
            origin: Vec3::ZERO,
            across: Vec3::new(card.width, 0.0, 0.0),
            along: Vec3::new(0.0, card.length().max(1e-4), 0.0),
            depth: card.depth,
        };
        let members = straightened(card, strands)?;
        if members.is_empty() {
            continue;
        }
        let maps: HairCardMaps = bake_hair_card(&members, &flat, cell[0], cell[1])?;
        let (x0, y0) = ((index % set.cols) * cell[0], (index / set.cols) * cell[1]);
        blit(&mut atlas.alpha, &maps.alpha, x0, y0);
        blit(&mut atlas.depth, &maps.depth, x0, y0);
        blit(&mut atlas.root, &maps.root, x0, y0);
        blit(&mut atlas.id, &maps.id, x0, y0);
        blit(&mut atlas.flow, &maps.flow, x0, y0);
    }
    Ok(atlas)
}

/// A PNG of `map` (values `v · scale + bias`, clamped, as 8 bits).
fn map_png(map: &BakeMap, scale: f32, bias: f32) -> Result<Vec<u8>> {
    let samples = map.data.iter().map(|&v| ((v * scale + bias).clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let image = Image::new(Format::Png, map.width, map.height, map.channels, ColorSpace::Data, Samples::U8(samples))?;
    Ok(png::encode(&image)?)
}

/// The base colour PNG: white with the coverage as alpha (sRGB).
fn base_png(alpha: &BakeMap) -> Result<Vec<u8>> {
    let samples = alpha.data.iter().flat_map(|&a| [255, 255, 255, (a.clamp(0.0, 1.0) * 255.0).round() as u8]).collect();
    let image = Image::new(Format::Png, alpha.width, alpha.height, 4, ColorSpace::Srgb, Samples::U8(samples))?;
    Ok(png::encode(&image)?)
}

/// Writes card LODs (finest first) with their atlases as one glTF binary (`.glb`).
pub fn export_cards_glb(lods: &[(&CardSet, &CardAtlas)]) -> Result<Vec<u8>> {
    if lods.is_empty() {
        return Err(GroomToolError::Invalid("no card LOD to export"));
    }
    let mut b = Builder::new("HumanArt hair cards");
    let mut nodes = Vec::with_capacity(lods.len());
    for (k, (set, atlas)) in lods.iter().enumerate() {
        let m = &set.mesh;
        if m.indices.is_empty() || m.positions.len() != m.normals.len() || m.positions.len() != m.uvs.len() {
            return Err(GroomToolError::Invalid("card mesh is empty or its attributes differ in length"));
        }
        let pos: Vec<f32> = m.positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let nrm: Vec<f32> = m.normals.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let uv: Vec<f32> = m.uvs.iter().flat_map(|p| [p.x, p.y]).collect();
        let pos = b.accessor_f32(&pos, ElementType::Vec3, Some(Target::ArrayBuffer))?;
        let nrm = b.accessor_f32(&nrm, ElementType::Vec3, Some(Target::ArrayBuffer))?;
        let uv = b.accessor_f32(&uv, ElementType::Vec2, Some(Target::ArrayBuffer))?;
        let idx = b.accessor_indices(&m.indices)?;
        let base = b.image(&base_png(&atlas.alpha)?, "image/png", Some(&format!("hair_lod{k}_base")))?;
        let extra = [
            ("depth", map_png(&atlas.depth, 1.0, 0.0)?),
            ("root", map_png(&atlas.root, 1.0, 0.0)?),
            ("id", map_png(&atlas.id, 1.0, 0.0)?),
            ("flow", map_png(&atlas.flow, 0.5, 0.5)?),
        ];
        let mut extra_images = Vec::with_capacity(extra.len());
        for (name, bytes) in &extra {
            extra_images.push((*name, b.image(bytes, "image/png", Some(&format!("hair_lod{k}_{name}")))?));
        }
        let d = &mut b.document;
        let base_texture = d.textures.len();
        d.textures.push(Texture { source: Some(base), name: Some(format!("hair_lod{k}_base")), ..Texture::default() });
        for (name, image) in extra_images {
            d.textures.push(Texture {
                source: Some(image),
                name: Some(format!("hair_lod{k}_{name}")),
                ..Texture::default()
            });
        }
        let material = d.materials.len();
        d.materials.push(Material {
            name: Some(format!("hair_cards_lod{k}")),
            pbr: Pbr {
                base_color_texture: Some(TextureInfo::new(base_texture)),
                metallic_factor: 0.0,
                ..Pbr::default()
            },
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Material::default()
        });
        let mesh = d.meshes.len();
        d.meshes.push(Mesh {
            name: Some(format!("hair_cards_lod{k}")),
            primitives: vec![Primitive {
                attributes: vec![("POSITION".into(), pos), ("NORMAL".into(), nrm), ("TEXCOORD_0".into(), uv)],
                indices: Some(idx),
                material: Some(material),
                ..Primitive::default()
            }],
            ..Mesh::default()
        });
        nodes.push(d.nodes.len());
        d.nodes.push(Node { name: Some(format!("hair_cards_lod{k}")), mesh: Some(mesh), ..Node::default() });
    }
    b.document.scenes.push(Scene { nodes, ..Scene::default() });
    b.document.scene = Some(0);
    Ok(b.finish()?.to_glb()?)
}
