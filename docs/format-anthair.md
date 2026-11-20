# The HumanArt strand file (`.anthair`) and the groom guide blob

The `.anthair` file holds a generated groom's strands `[IO-12]` `[HAIR-01]`: one chunk per
per-strand attribute, each stored raw or LZ-compressed (`humanart_io::lz`, `docs/format-lz.md`).
The guide blob holds the guide strands a document groom references `[DOC-08]`. Implementation:
`crates/model/src/hair/format.rs` (`humanart_model::hair::format`: `write` / `read`,
`write_guides` / `read_guides`); this page is taken from that module's doc comments and code.

All integers and floats are little-endian (`f32` is IEEE 754 binary32, `f16` binary16), written
and read explicitly. Offsets are byte offsets from the start of the structure being described.

## `.anthair`, version 1

### Header (24 bytes)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | the bytes `HAIR` |
| 4 | 2 | version | `1`; any other value is refused |
| 6 | 2 | flags | `0` (no flag is defined); any other value is refused |
| 8 | 4 | strands *n* | at most `MAX_STRANDS` (2^22) |
| 12 | 4 | points *p* | control points per strand, 2..=32 (`MAX_STRAND_POINTS`) |
| 16 | 4 | chunk count | at most `MAX_CHUNKS` (64) |
| 20 | 4 | reserved | `0`; any other value is refused |

The chunks follow back to back, each a 16-byte header and then its stored bytes. The file ends
after the last chunk; bytes after it are refused.

### Chunk header (16 bytes)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | tag | four ASCII bytes, see below |
| 4 | 1 | codec | `0` raw, `1` one LZ frame (`docs/format-lz.md`); others are refused |
| 5 | 3 | reserved | `0` |
| 8 | 4 | raw length | length of the decoded content; must equal the length the table below gives for *n* and *p* |
| 12 | 4 | stored length | length of the stored bytes that follow; for a raw chunk it equals the raw length |

The writer stores a chunk LZ-compressed only when compression is asked for and the frame is smaller
than the raw content.

### Chunks (decoded content)

Arrays of *n* entries unless noted.

| Tag | Content | Raw length |
|---|---|---|
| `ROOT` | root triangle `u32` x *n*, then barycentric `(u, v)` `f32` x 2 x *n* | 12 *n* |
| `OFFS` | point offsets in the root frame, `f16` x, y, z, *n* x *p* points | 6 *n p* |
| `LENG` | strand length, metres, `f32` | 4 *n* |
| `WIDR` | root diameter, metres, `f32` | 4 *n* |
| `WIDT` | tip diameter, metres, `f32` | 4 *n* |
| `COLR` | colour variation id, `u16` | 2 *n* |
| `CURL` | curl phase, fraction of a turn, `f32` (see the note below) | 4 *n* |
| `CLMP` | clump id (index of the clump's centre strand), `u32` | 4 *n* |
| `STIF` | stiffness hint, 0 (limp) to 1 (stiff), `f32` | 4 *n* |
| `GREY` | grey level, 0 pigmented to 255 unpigmented, `u8` | *n* |

Every one of the ten tags is required exactly once; chunks may come in any order; a chunk with an
unknown tag is skipped (later versions may add attributes). The writer writes the ten chunks in the
order of the table.

Offsets are in the root frame of `humanart_model::hair::RootFrame`: the world position of the
local offset `(x, y, z)` is `origin + t x + b y + n z`, where `origin` is the root on the surface,
`n` the surface normal, `t` the tangent (UV `u` direction) and `b = n x t`. A file therefore needs
the skin it was made on (triangles, positions, normals, UVs) to place its strands in space.

**Curl phase unit.** A fraction of a turn in `[0, 1)`: the CPU generator and
`shaders/groom/generate.wgsl` add it to `frequency * s` and multiply the fractional part by 2π.

### Reader checks, in order `[ROB-01]`

1. Magic, version, flags, strand count (<= 2^22), point count (2..=32), chunk count (<= 64),
   reserved word.
2. The total decoded size of the ten required chunks, computed in `u64` from *n* and *p*, against
   `limits.max_bytes`, before anything is allocated.
3. Per chunk: the header fits the input, the reserved bytes are 0, the stored bytes fit the input;
   unknown tags are skipped; a known tag twice is refused; its raw length must equal the expected
   length; a raw chunk's stored length must equal its raw length; an LZ chunk is decoded with its
   raw length as the limit and must decode to exactly that length; an unknown codec is refused.
4. No bytes after the last chunk; every required tag present.
5. The strands themselves: finite barycentric roots, then `Strands::check` (array lengths, clump
   ids below *n*, finite lengths, widths, curl phases, stiffness and offsets). Roots are checked
   against a surface only by a caller that has one.

Format errors carry the byte offset of the field at fault. Mutation fuzz:
`format::tests` (4000 cases, fixed seeds).

## Guide blob (`HAGD`), version 1

The guides of a document groom (`Groom::guides`) and the payload of a `groom_guide_edit` stroke.
They round-trip bit for bit (no resampling).

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | the bytes `HAGD` |
| 4 | 2 | version | `1` |
| 6 | 2 | flags | `0` |
| 8 | 4 | guides *n* | at most `MAX_GUIDES` (2^18) |
| 12 | 4 | points *p* | 2..=64 (`MAX_GUIDE_POINTS`) |
| 16 | 1 | codec | `0` raw, `1` one LZ frame |
| 17 | 3 | reserved | `0` |
| 20 | 4 | raw length | must equal 12 *n* + 12 *n p* |
| 24 | 4 | stored length | the stored content that follows; nothing comes after it |
| 28 | 4 | reserved | `0` |
| 32 | | content | see below |

Content (decoded): root triangle `u32` x *n*, barycentric `(u, v)` `f32` x 2 x *n*, then offsets
`f32` x, y, z x *n* x *p* in the root frame (the first point of each guide is the root). Unlike the
strand file, guide offsets are `f32`.

The reader checks magic, version, flags, guide count, point count and reserved bytes, then the raw
length against the guide count and against `limits.max_bytes`, then that the stored bytes fit and
end the input, then decodes (raw length = stored length, or LZ to exactly the raw length) and
builds the guides through `Guides::from_raw`. Mutation fuzz: 3000 cases.
