# The HumanArt template package (`.antmpl`)

The package that holds the template `[IO-21]` `[MOD-02]`: LOD0 mesh, UV tiles, regions, landmarks,
skeleton and joint regressor, skin weights, sparse delta bases, expression and corrective
definitions, further LODs, and a JSON metadata section with the licence and provenance manifest and
extra data (parameter registry data, rig logic). The procedural generator and the Template Adapter
both write it; the rest of the system reads only this. Implementation: `crates/model/src/package/`
(`humanart_model::package`, `write_package` / `read_package`).

All integers and floats are little-endian (`f32` is IEEE 754 binary32, `f16` binary16). Offsets in
the tables are byte offsets from the start of the structure being described. The writer is
deterministic: one template always gives the same bytes, and writing what was read reproduces the
file byte for byte.

## Header (32 bytes)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | the bytes `48 41 54 50` (`HATP`) |
| 4 | 2 | major version | `1`; any other value is refused as unsupported |
| 6 | 2 | minor version | written `0`; any value is read (minor versions only add optional sections) |
| 8 | 4 | section count *N* | at most 64 |
| 12 | 4 | flags | `0`; any other value is refused |
| 16 | 8 | file length | must equal the length of the file |
| 24 | 8 | table checksum | see below |
| 32 | 48 *N* | section table | *N* entries |

The **table checksum** is MurmurHash3 x64_128 (seed 0, `humanart_core::hash`) of the header bytes
0..24 followed by the section table, folded to 64 bits as `h1 XOR rotate_left(h2, 32)`.

## Section table entry (48 bytes)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | tag | four ASCII bytes, unique in the file |
| 4 | 2 | section version | `1` for every section below; a known tag with another version is refused |
| 6 | 2 | flags | bit 0: compressed; bit 1: required; other bits `0` |
| 8 | 8 | offset | from the start of the file; see placement |
| 16 | 8 | stored length | bytes in the file |
| 24 | 8 | raw length | bytes of the payload; equals the stored length when not compressed |
| 32 | 16 | hash | MurmurHash3 x64_128 of the stored bytes, as big-endian `h1` then big-endian `h2` |

**Placement.** The first section starts at the first multiple of 16 at or after the end of the
table; each further section at the first multiple of 16 at or after the end of the previous one.
Padding bytes are `0`. The file ends exactly at the end of the last section. Sections may appear in
any order; the writer uses the order of the list below.

**Compression.** A compressed section's stored bytes are one HumanArt LZ frame
(`docs/format-lz.md`) whose content is the payload. The writer compresses a section when that makes
it smaller.

**Unknown sections.** A reader skips a section whose tag it does not know unless bit 1 (required)
is set, in which case it refuses the file. Every section of this version is written as required.
All ten sections below must be present.

## Common encodings

- `str`: `u16` byte length, then that many bytes of UTF-8. Names (regions, joints, ...) are stable
  ids: 1 to 64 bytes of ASCII letters, digits, `.`, `-`, `_`, starting with a letter or digit, unique
  within their table.
- `vec3`: three `f32` (x, y, z). `quat`: four `f32` (x, y, z, w), a unit quaternion.
- `parent`: `u16` index of a lower-indexed entry of the same table, or `FFFF` for none.
- `side`: `u8`, 0 centre, 1 left (+X), 2 right (−X).
- Counts are `u32` unless stated. *V* is the vertex count of the mesh the section belongs to.

Every count is checked against its limit (below and in `PackageLimits`) and every array against the
bytes left in the payload before anything is allocated. A payload must be consumed exactly.

## Sections

### `META`: metadata (JSON)

UTF-8 JSON (RFC 8259, `humanart_io::json`), compact, members in this order:

| Key | Value |
|---|---|
| `format` | the string `humanart-template` |
| `name` | template name (a stable name) |
| `version` | content version, integer 0..2^32 |
| `generator` | what produced the template (≤ 4096 bytes) |
| `licence` | licence of the template (≤ 4096 bytes) |
| `provenance` | array (≤ 1024) of `{"item", "source", "licence"}` strings (≤ 4096 bytes each) |
| `extra` | object: further data by key (registry data, rig logic); copied verbatim |

Unknown top-level keys are ignored.

### `MESH`: LOD0 mesh

| Field | Encoding |
|---|---|
| vertex count *V* | `u32`, ≤ 2^24 |
| face count *F* | `u32`, ≤ 2^24 |
| corner count *K* | `u32` |
| part count *P* | `u32`, ≤ 256 |
| positions | *V* × `vec3`, metres, finite |
| UVs | *V* × 2 `f32`, tile-local, origin top-left, V down, finite |
| weld keys | *V* × `u32`: the lowest vertex with the same position (itself if not duplicated) |
| normal keys | *V* × `u32`: the lowest vertex sharing the normal; same weld key |
| face offsets | (*F* + 1) × `u32`: from 0, each face 3 to 64 corners, last = *K* |
| face corners | *K* × `u32` vertex indices, counter-clockwise from outside; no weld key twice in a face |
| face tiles | *F* × `u16` UV tile index |
| parts | *P* × { `str` name, `u32` first face, `u32` face count (≥ 1), `u8` closed (0/1) }, contiguous, covering all faces |

Duplicated vertices (UV seams, hard edges) have bit-identical positions; a weld key or normal key is
its own key.

### `TILE`: UV tiles

`u32` count (≤ 256), then per tile: `str` name, `u32` resolution (1..65536 texels a side), `f32`
target texel density (texels per millimetre, finite, ≥ 0). Every face tile index is below the count.

### `REGN`: regions

| Field | Encoding |
|---|---|
| region count *R* | `u32`, ≤ 4096 |
| regions | *R* × { `str` name, `parent`, `side` } |
| vertex count | `u32`, equals *V* |
| vertex regions | *V* × `u16` region index |
| soft flag | `u8`: 0 none, 1 present |
| if present: entry count *E* | `u32` |
| offsets | (*V* + 1) × `u32`, from 0, non-decreasing, ≤ 16 entries per vertex, last = *E* |
| entry regions | *E* × `u16` |
| entry weights | *E* × `f32` in [0, 1] |

### `LMRK`: landmarks

`u32` count (≤ 4096), then per landmark: `str` name, `u8` group (0 face, 1 body), 3 × `u32` vertices
(corners of one face), 3 × `f32` barycentric weights (sum 1).

### `SKEL`: skeleton and joint regressor

`u32` joint count *J* (≤ 1024), then per joint: `str` name, `parent`, `u8` kind (0 deform, 1 twist,
2 helper, 3 socket), rest transform relative to the parent (`vec3` translation, `quat` rotation,
`vec3` scale > 0), limits (`vec3` min, `vec3` max: Euler Y-X-Z radians, min ≤ max, |·| ≤ 2π).

Then the regressor: `u32` component count *C* (equals the `DLTA` component count), `u32` entry count
*E*, (*C* + 1) × `u32` offsets (from 0, non-decreasing, last = *E*), *E* × `u16` joint (strictly
increasing within a component), *E* × `vec3` world-space joint position offset per unit weight of
the component.

### `SKIN`: skin weights

`u8` influences per vertex *k* (1..8; 0 only when *J* = 0), `u32` vertex count (equals *V*, written
0 when *k* = 0), *V k* × `u16` joint, *V k* × `f32` weight (in [0, 1], per vertex decreasing, sum 1),
`u32` count and that many `u16` regions skinned with dual quaternions (sorted, unique).

### `DLTA`: sparse delta bases `[EVAL-02]`

| Field | Encoding |
|---|---|
| component count *C* | `u32`, ≤ 65536 |
| entry count *E* | `u32`, ≤ 2^28 |
| components | *C* × { `str` name, `u8` kind, `u16` bounding region } |
| offsets | (*C* + 1) × `u32`, from 0, non-decreasing, last = *E* |
| vertex indices | *E* × `u32`, strictly increasing within a component |
| displacements | *E* × 3 `f16` (x, y, z), metres per unit weight, finite |

Kinds: 0 global identity, 1 regional identity, 2 semantic, 3 asymmetry, 4 expression, 5 combination
corrective, 6 pose corrective. Components are sorted by bounding region (non-decreasing). With no
scan data `[ID-07]` kinds 0, 1 and 3 are simply absent.

### `EXPR`: expressions and correctives

- `u32` expression count (≤ 4096), per expression: `str` name, `u8` kind (0 action unit, 1 viseme,
  2 tongue, 3 support), `side`, `u16` FACS number (0 = none), `u32` component (an expression delta).
- `u32` combination count (≤ 4096), per combination: `str` name, `u8` op (0 product, 1 min), `u8`
  input count (2..4), that many `u32` distinct expression indices, `u32` component (a combination
  corrective delta).
- `u32` pose corrective count (≤ 4096), per corrective: `str` name, `u16` joint, `quat` target
  rotation relative to rest, `f32` radius in (0, π], `u32` component (a pose corrective delta).
  Activation: `(1 − (a / radius)²)²` for the angle `a` to the target below the radius, else 0,
  unless a pose driver (`PDRV`) owns the corrective.

### `PDRV`: pose drivers (optional, todo 1.3)

Written after `LODS`, flagged required, and only when the template has pose drivers (templates
without any are byte-identical to before). `u32` driver count (≤ 4096), per driver: `str` name,
`u16` joint, `u8` metric (0 swing, 1 twist, 2 full; twist is about the joint frame's Y axis, the
bone), `f32` kernel radius in (0, π], `u32` corrective count (1..32), that many `u32` distinct pose
corrective indices, each corrective driving the same joint and owned by no other driver. The
correctives' targets are the driver's example poses; the rest pose is an implicit example with every
output 0. Activations: Gaussian radial basis interpolation `exp(−(d/r)²)` of the examples, clamped to
[0, 1] and scaled down when they sum above 1 (`humanart_model::expression::pose_space`).

### `LODS`: LOD1 and further `[MOD-05]`

`u32` level count (≤ 4), then per level: `f32` declared error (metres, ≥ 0), `f32` switch size
(fraction of screen height in (0, 1], strictly decreasing from level to level), `f32` hysteresis
([0, 0.5]), a mesh encoded exactly like `MESH` (its own *V'*), *V'* × 3 `u32` LOD0 source vertices,
*V'* × 3 `f32` source weights (each in [0, 1], sum 1 within 10⁻³).

## Reader checks, in order

1. File size against `PackageLimits::max_file_bytes`; magic; major version; section count ≤ 64;
   header flags; file length; table checksum.
2. Per entry: known flag bits, unique tag, canonical placement, zero padding, stored = raw length
   when uncompressed, raw length ≤ `max_section_bytes`, hash of the stored bytes. No trailing bytes.
3. Per known section: version 1, decompression to exactly the raw length, decoding with every count
   checked against limits and remaining bytes, no trailing bytes. Unknown required section: refused.
   Missing section: refused.
4. The decoded template is checked as a whole (`Template::new`): every index in range, orders,
   names, finite values. Quality checks (watertightness, UV overlap, ...) are a separate report
   (`humanart_model::validate`), not a load failure.
