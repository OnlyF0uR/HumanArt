"""Dev-time fixture source [DEP-04]: writes a deterministic HDR test picture as raw planar float.

usage: python gen_src.py <width> <height> <planes: gbr|gbra|gray> <out.raw>
Planes are written in ffmpeg's order for gbrpf32le / gbrapf32le / grayf32le (G, B, R[, A] or Y),
little-endian f32, rows top to bottom. Values span about 2^-9 .. 2^9 with a black corner, exact
ramps and pseudo-random noise from a fixed LCG (no library randomness).
"""
import struct
import sys


def pixel(x, y, w, h, state):
    state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
    noise = (state[0] >> 8) / float(1 << 23)
    if x < 3 and y < 3:
        return (0.0, 0.0, 0.0, 0.0)
    r = 2.0 ** ((x - w / 2.0) / (w / 18.0)) * (1.0 + y / float(h))
    g = (x * y) / 50.0 + noise * 0.25
    b = 0.001 * (1 + (x * 7 + y * 13) % 32) if (x // 5 + y // 4) % 2 else 4.0 * noise * noise
    a = min(1.0, (x + y) / float(w + h - 2))
    return (r, g, b, a)


def main():
    w, h, planes, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    state = [20261119]
    px = [[pixel(x, y, w, h, state) for x in range(w)] for y in range(h)]
    order = {"gbr": [1, 2, 0], "gbra": [1, 2, 0, 3], "gray": [1]}[planes]
    with open(out, "wb") as f:
        for c in order:
            for row in px:
                f.write(struct.pack("<%df" % w, *[struct.unpack("<f", struct.pack("<f", p[c]))[0] for p in row]))


main()
