"""Makes the DEFLATE / zlib fixtures of humanart-io (dev-time only, [DEP-04]).

Run from anywhere with Python 3 (standard library only):

    python -I crates/io/tests/fixtures/deflate/make_fixtures.py

The input is generated here from a fixed seed, so it carries no third-party content. Every output is
written next to this script; the Rust tests decode them and compare against input.bin.
"""

import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield state >> 33


def make_input():
    rng = lcg(20261119)
    words = [b"skin", b"pore", b"albedo", b"melanin", b"strand", b"cortex", b"specular", b"the",
             b"of", b"a", b"subsurface", b"scatter", b"normal", b"roughness", b"eyelid", b"iris"]
    out = bytearray()
    # Text-like: many medium matches at short and long distances.
    while len(out) < 30000:
        out += words[next(rng) % len(words)]
        out += b" " if next(rng) % 9 else b".\n"
    # Noise: incompressible, forces literal-heavy blocks.
    out += bytes(next(rng) & 0xFF for _ in range(20000))
    # A long run (distance-1 matches) and a slow ramp (2-byte periods).
    out += b"\x00" * 9000
    out += bytes((i // 2) & 0xFF for i in range(6000))
    # A repeat of the first text far back (distances near the 32 KiB window).
    out += out[:5000]
    return bytes(out)


def write(name, data):
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(data)


def compress(data, level, wbits=15, strategy=zlib.Z_DEFAULT_STRATEGY):
    c = zlib.compressobj(level, zlib.DEFLATED, wbits, 9, strategy)
    return c.compress(data) + c.flush()


def main():
    data = make_input()
    write("input.bin", data)
    write("zlib_l0.zz", compress(data, 0))
    write("zlib_l1.zz", compress(data, 1))
    write("zlib_l6.zz", compress(data, 6))
    write("zlib_l9.zz", compress(data, 9))
    write("zlib_w9.zz", compress(data, 9, wbits=9))
    write("zlib_fixed.zz", compress(data, 6, strategy=zlib.Z_FIXED))
    write("zlib_huffman.zz", compress(data, 6, strategy=zlib.Z_HUFFMAN_ONLY))
    write("zlib_rle.zz", compress(data, 6, strategy=zlib.Z_RLE))
    write("raw_l6.deflate", compress(data, 6, wbits=-15))
    # A bomb: 4 MiB of zeros in about 4 KiB.
    write("zeros_4m.zz", compress(b"\x00" * (4 << 20), 9))
    # A preset dictionary, which the decoder refuses as unsupported.
    c = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, b"melanin cortex")
    write("zlib_dict.zz", c.compress(b"melanin cortex melanin") + c.flush())
    for name in sorted(os.listdir(HERE)):
        if name.endswith((".zz", ".deflate")):
            print(name, os.path.getsize(os.path.join(HERE, name)))
    print("input.bin", len(data), "adler32 %08x" % zlib.adler32(data), "zlib", zlib.ZLIB_VERSION)


if __name__ == "__main__":
    main()
