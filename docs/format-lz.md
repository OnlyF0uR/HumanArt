# The HumanArt LZ format

The in-house compression codec `[DEP-05]` used for blobs `[DOC-22]` `[IO-30]`, for KTX2
supercompression `[IO-04]` and inside chunked containers. It is a byte-oriented LZ77 codec with no
entropy stage: fast to compress, faster to decompress, a modest ratio. Implementation:
`crates/io/src/lz/` (`humanart_io::lz`, feature `lz`).

All integers are little-endian. Offsets in this document are byte offsets from the start of the
structure being described.

## Frame

A frame holds one piece of content of known size, split into blocks of a fixed size.

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | the bytes `48 41 4C 5A` (`HALZ`) |
| 4 | 1 | version | `1`; any other value is refused as unsupported |
| 5 | 1 | block size | log2 of the block size, 16 to 22 (64 KiB to 4 MiB) |
| 6 | 2 | reserved | `0`; any other value is refused |
| 8 | 8 | content size | the size of the decompressed content in bytes |
| 16 | ... | blocks | `ceil(content size / block size)` blocks (none for empty content) |
| end − 8 | 8 | checksum | see below |

Nothing may follow the checksum. The *length* of block *i* (its decompressed size) is the block size,
except for the last block, which holds what remains of the content size; it is never stored, only
derived.

Each block is a 4-byte *block word* followed by its payload:

| Bits of the block word | Meaning |
|---|---|
| 31 | `1`: the payload is the block's bytes stored as they are; `0`: the payload is an LZ block (below) |
| 0 to 30 | the payload size in bytes |

A stored block's payload size equals the block length. A compressed block's payload size is at most
`len + len / 255 + 16` for a block length `len`. The encoder stores a block whenever compressing it
does not make it smaller, so a frame is at most 24 bytes plus 4 bytes per block larger than its
content.

**Checksum:** the 128-bit MurmurHash3 x64_128 (seed 0) of the whole decompressed content, `h1` and
`h2` folded to 64 bits as `h1 XOR rotate_left(h2, 32)` (`humanart_core::hash::Hash128::fold64`),
written as a u64.

Blocks are independent: a match never reaches outside its own block, so blocks can be decoded in
any order or in parallel.

## Block

A compressed block is a sequence of *sequences*. Each sequence is:

| Field | Size | Meaning |
|---|---|---|
| token | 1 | high nibble: literal count `L` (0 to 15); low nibble: match length code `M` (0 to 15) |
| literal length bytes | 0 or more | present when `L = 15`, see *lengths* |
| literals | literal count | bytes copied to the output as they are |
| match offset | 2 | u16, 1 to 65 535: how far back the match starts in the output |
| match length bytes | 0 or more | present when `M = 15`, see *lengths* |

The **last sequence** of a block ends right after its literals: it has no offset and no match. A
block is complete exactly when its input is used up after a run of literals; it must then have
produced exactly the block length.

**Lengths.** The literal count is `L`, or, when `L = 15`, 15 plus the sum of the following length
bytes: bytes are read and added while each is 255, and the first byte below 255 is the last one
(`15 + 255 + 255 + 3` is the bytes `FF FF 03` after a nibble of 15). The match length is `M + 4` in
the same way (the shortest match is 4 bytes).

**Matches.** A match copies *match length* bytes, one at a time, from *offset* bytes before the
current end of the output; it may overlap what it writes (offset 1 with length 10 repeats the last
byte ten times).

The sequence layout is that of the LZ4 block format; interoperability with other LZ4
implementations is neither claimed nor tested.

## What a decoder refuses

- a frame shorter than its header, a wrong magic, an unknown version, a block size outside 16 to 22,
  non-zero reserved bytes;
- a content size over the caller's limit (`Limits::max_bytes`), or more than 256 times the frame
  length (no valid frame expands further), checked **before** any output is allocated `[ROB-02]`;
- a block word or payload that runs past the input, a stored block whose size is not its block
  length, a compressed payload larger than the bound above;
- in a block: an offset of 0 or reaching before the start of the block, literals or a match that run
  past the block length, a length whose extra bytes run past the block length, input that ends
  inside a sequence or after a match, a block that produces less than its length;
- a checksum that does not match, and any byte after the checksum.

On any error the caller's output buffer is left as it was.

## Encoder

The reference encoder (`lz::Encoder`) is greedy and single-pass: a 16 384-entry hash table of the
4-byte sequences seen so far in the block, extending each hit backwards and forwards, skipping ahead
faster the longer nothing matches. It always leaves the last 5 bytes of a block as literals and
starts no match within the last 12 bytes. Its output is deterministic: the same content and block
size always give the same frame.

## Measured speed

32 MiB inputs, release build, one thread, on the development PC (2026-11-19); compression with
256 KiB blocks, decompression including the checksum:

| Content | Ratio | Compress | Decompress |
|---|---|---|---|
| English text (the spec, repeated beyond the 64 KiB window) | 0.571 | 327 MB/s | 1 234 MB/s |
| Sparse bytes (60 % zeros) | 0.327 | 650 MB/s | 1 354 MB/s |
| Smooth `f32` mesh data | 1.000 (stored) | 2 933 MB/s | 4 276 MB/s |
| Random bytes | 1.000 (stored) | 2 923 MB/s | 4 256 MB/s |

Reproduce with
`cargo test -p humanart-io --features lz --release measure_speed -- --ignored --nocapture`.
