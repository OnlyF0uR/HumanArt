"""Writes the PLY test fixtures of humanart-io (dev-time only, Python standard library) [DEP-04].

One tetrahedron with every PLY scalar type, a list property and three elements, in the three
encodings. The bytes are laid out here independently of the Rust writer, so that the tests can check
the reader against them and the writer for byte-identical output.

Run from this directory: python -I make_fixtures.py
"""
import struct

HEADER = """ply
format {fmt} 1.0
comment made by make_fixtures.py for the HumanArt PLY tests
obj_info tetrahedron with every scalar type
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
property float u
property float v
property double quality
element face 4
property list uchar int vertex_indices
property short flags
element edge 3
property int vertex1
property uint vertex2
property char sign
property ushort weight
end_header
"""

POS = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
VERTICES = [
    (*p, *(c * 0.5 for c in p), *rgba, *uv, q)
    for p, rgba, uv, q in zip(
        POS,
        [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 128), (255, 255, 255, 255)],
        [(0, 0), (1, 0), (0, 1), (0.5, 0.25)],
        [0.1, 0.2, 12345.678, -3.5],
    )
]
FACES = [([0, 2, 1], 0), ([0, 1, 3], -1), ([0, 3, 2], 32767), ([1, 2, 3], -32768)]
EDGES = [(0, 1, 1, 0), (1, 2, -1, 65535), (2, 3, -128, 7)]
VERTEX_TYPES = "ffffffBBBBffd"


def text(value):
    """The shortest decimal form, without a trailing '.0' (as Rust's Display writes it)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def ascii_data():
    rows = [" ".join(text(v) for v in row) for row in VERTICES]
    rows += [" ".join([str(len(f))] + [str(i) for i in f] + [str(flags)]) for f, flags in FACES]
    rows += [" ".join(str(v) for v in row) for row in EDGES]
    return ("\n".join(rows) + "\n").encode()


def binary_data(order):
    out = b""
    for row in VERTICES:
        out += struct.pack(order + VERTEX_TYPES, *row)
    for f, flags in FACES:
        out += struct.pack(order + "B" + "i" * len(f) + "h", len(f), *f, flags)
    for row in EDGES:
        out += struct.pack(order + "iIbH", *row)
    return out


for name, fmt, data in [
    ("tetra_ascii.ply", "ascii", ascii_data()),
    ("tetra_le.ply", "binary_little_endian", binary_data("<")),
    ("tetra_be.ply", "binary_big_endian", binary_data(">")),
]:
    with open(name, "wb") as f:
        f.write(HEADER.format(fmt=fmt).encode() + data)
