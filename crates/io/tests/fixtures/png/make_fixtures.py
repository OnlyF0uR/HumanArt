"""Makes the PNG fixtures of humanart-io (dev-time only, [DEP-04]).

    python -I crates/io/tests/fixtures/png/make_fixtures.py          # write the fixtures
    python -I crates/io/tests/fixtures/png/make_fixtures.py --check [DIR]  # decode with ffmpeg, compare
    python -I crates/io/tests/fixtures/png/make_fixtures.py --ffmpeg       # the ff_* fixtures

A small PNG writer independent of the Rust code: every colour type and bit depth, Adam7, a random
filter per row, palettes with tRNS, split IDAT chunks and the colour chunks. The pixels come from a
fixed-seed LCG (no third-party content). Beside every `<name>.png` it writes `<name>.bin`, what the
Rust decoder must produce:

    width u32 LE, height u32 LE, channels u8, bits u8 (8 or 16), colour space u8
    (0 sRGB, 1 linear, 2 normal, 3 data), then the samples (u8, or u16 LE), rows top to bottom.

Bit depths below 8 are scaled to 8 bits (v * 255 / (2^d - 1)); a palette is expanded to RGB, or
RGBA when tRNS is present; grey and RGB tRNS add an alpha channel (0 on the key colour, else max).
`--check` decodes every fixture with ffmpeg into the matching raw format and compares.
"""

import os
import struct
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


class Lcg:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 33

    def below(self, n):
        return self.next() % n


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def pack_row(samples, depth):
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    if depth == 8:
        return bytes(samples)
    out = bytearray()
    per = 8 // depth
    for i in range(0, len(samples), per):
        byte = 0
        group = samples[i:i + per]
        for j, s in enumerate(group):
            byte |= s << (8 - depth * (j + 1))
        out.append(byte)
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(kind, row, prev, bpp):
    out = bytearray(len(row))
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i] if prev is not None else 0
        c = prev[i - bpp] if prev is not None and i >= bpp else 0
        pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][kind]
        out[i] = (x - pred) & 0xFF
    return bytes([kind]) + bytes(out)


def image_data(pixels, w, h, ch, depth, interlace, rng):
    """pixels[y][x] is a list of ch raw samples; returns the filtered, unconcatenated scanlines."""
    bpp = max(1, ch * depth // 8)
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        pw = (w - x0 + dx - 1) // dx if w > x0 else 0
        ph = (h - y0 + dy - 1) // dy if h > y0 else 0
        if pw == 0 or ph == 0:
            continue
        prev = None
        for py in range(ph):
            y = y0 + py * dy
            samples = []
            for px in range(pw):
                samples += pixels[y][x0 + px * dx]
            row = pack_row(samples, depth)
            raw += filter_row(rng.below(5), row, prev, bpp)
            prev = row
    return bytes(raw)


def make(name, w, h, ctype, depth, interlace=False, trns=None, palette_len=None, extra=b"", idat_split=0,
         colour_space=0, seed=1):
    rng = Lcg(seed * 7919 + w * 31 + h + ctype * 1000 + depth * 100 + interlace)
    ch = CHANNELS[ctype]
    maxv = (1 << depth) - 1
    palette = None
    if ctype == 3:
        palette_len = palette_len or (1 << depth)
        palette = [(rng.below(256), rng.below(256), rng.below(256)) for _ in range(palette_len)]
        pixels = [[[rng.below(palette_len)] for _ in range(w)] for _ in range(h)]
    else:
        pixels = [[[rng.below(maxv + 1) for _ in range(ch)] for _ in range(w)] for _ in range(h)]
        # Make sure the tRNS key colour occurs.
        if trns is not None and w * h > 2:
            pixels[0][1] = list(trns)
            pixels[h - 1][w - 1] = list(trns)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 1 if interlace else 0))
    png += extra
    if palette is not None:
        png += chunk(b"PLTE", b"".join(bytes(p) for p in palette))
    if trns is not None:
        if ctype == 3:
            png += chunk(b"tRNS", bytes(trns))
        else:
            png += chunk(b"tRNS", b"".join(struct.pack(">H", v) for v in trns))
    stream = zlib.compress(image_data(pixels, w, h, ch, depth, interlace, rng), 9)
    if idat_split:
        png += chunk(b"IDAT", b"")
        for i in range(0, len(stream), idat_split):
            png += chunk(b"IDAT", stream[i:i + idat_split])
    else:
        png += chunk(b"IDAT", stream)
    png += chunk(b"tEXt", b"Comment\x00made by make_fixtures.py")
    png += chunk(b"IEND", b"")

    # The expected decoded image.
    out_bits = 16 if depth == 16 else 8
    out_max = (1 << out_bits) - 1
    samples = []
    out_ch = None
    for y in range(h):
        for x in range(w):
            p = pixels[y][x]
            if ctype == 3:
                px = list(palette[p[0]])
                if trns is not None:
                    px.append(trns[p[0]] if p[0] < len(trns) else 255)
            else:
                px = [v * 255 // maxv for v in p] if depth < 8 else list(p)
                if trns is not None:
                    px.append(0 if list(p) == list(trns) else out_max)
            out_ch = len(px)
            samples += px
    fmt = "<%dH" % len(samples) if out_bits == 16 else None
    body = struct.pack(fmt, *samples) if fmt else bytes(samples)
    expected = struct.pack("<IIBBB", w, h, out_ch, out_bits, colour_space) + body
    with open(os.path.join(HERE, name + ".png"), "wb") as f:
        f.write(png)
    with open(os.path.join(HERE, name + ".bin"), "wb") as f:
        f.write(expected)


def all_fixtures():
    # Every legal colour type and bit depth, plain and interlaced.
    for ctype, depths in [(0, [1, 2, 4, 8, 16]), (2, [8, 16]), (3, [1, 2, 4, 8]), (4, [8, 16]), (6, [8, 16])]:
        for depth in depths:
            for interlace in (False, True):
                name = "c%d_d%d%s" % (ctype, depth, "_i" if interlace else "")
                yield name, dict(w=13, h=11, ctype=ctype, depth=depth, interlace=interlace)
    # Adam7 with empty passes, and a single pixel.
    for w, h in [(1, 1), (2, 3), (5, 1), (1, 9), (9, 2)]:
        yield "rgba8_i_%dx%d" % (w, h), dict(w=w, h=h, ctype=6, depth=8, interlace=True)
    yield "grey1_i_3x5", dict(w=3, h=5, ctype=0, depth=1, interlace=True)
    # tRNS for every colour type that takes it.
    yield "c0_d2_trns", dict(w=13, h=11, ctype=0, depth=2, trns=(1,))
    yield "c0_d8_trns_i", dict(w=13, h=11, ctype=0, depth=8, interlace=True, trns=(77,))
    yield "c0_d16_trns", dict(w=13, h=11, ctype=0, depth=16, trns=(4660,))
    yield "c2_d8_trns", dict(w=13, h=11, ctype=2, depth=8, trns=(10, 20, 30))
    yield "c2_d16_trns_i", dict(w=13, h=11, ctype=2, depth=16, interlace=True, trns=(1000, 2000, 3000))
    yield "c3_d4_trns", dict(w=13, h=11, ctype=3, depth=4, trns=(0, 128, 255, 7, 64))
    yield "c3_d8_short", dict(w=17, h=6, ctype=3, depth=8, palette_len=200, trns=(5, 6, 7))
    # Colour chunks and chunk layout.
    yield "srgb", dict(w=8, h=8, ctype=2, depth=8, extra=chunk(b"sRGB", b"\x00"), colour_space=0)
    yield "gama_linear", dict(w=8, h=8, ctype=2, depth=16, extra=chunk(b"gAMA", struct.pack(">I", 100000)),
                              colour_space=1)
    yield "gama_22", dict(w=8, h=8, ctype=0, depth=8, extra=chunk(b"gAMA", struct.pack(">I", 45455)),
                          colour_space=0)
    chrm = struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
    icc = chunk(b"iCCP", b"test profile\x00\x00" + zlib.compress(b"placeholder bytes: the decoder does not parse ICC"))
    yield "iccp_over_gama", dict(w=8, h=8, ctype=6, depth=8,
                                 extra=chunk(b"gAMA", struct.pack(">I", 100000)) + chunk(b"cHRM", chrm) + icc,
                                 colour_space=0)
    yield "split_idat", dict(w=31, h=17, ctype=6, depth=8, idat_split=7,
                             extra=chunk(b"prVt", b"private ancillary chunk"))


def main():
    for name, kw in all_fixtures():
        make(name, **kw)
    print("wrote", len(list(all_fixtures())), "fixtures")


def check(folder=HERE):
    """Decodes every `<name>.png` with a `<name>.bin` beside it in `folder` with ffmpeg."""
    fmts = {(1, 8): "gray", (2, 8): "ya8", (3, 8): "rgb24", (4, 8): "rgba",
            (1, 16): "gray16le", (2, 16): "ya16le", (3, 16): "rgb48le", (4, 16): "rgba64le"}
    ok = bad = 0
    names = sorted(n[:-4] for n in os.listdir(folder) if n.endswith(".png"))
    for name in names:
        if not os.path.exists(os.path.join(folder, name + ".bin")):
            continue
        with open(os.path.join(folder, name + ".bin"), "rb") as f:
            expected = f.read()
        w, h, ch, bits, cs = struct.unpack("<IIBBB", expected[:11])
        fmt = fmts[(ch, bits)]
        got = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(folder, name + ".png"), "-f", "rawvideo",
                              "-pix_fmt", fmt, "-"], capture_output=True).stdout
        if got == expected[11:]:
            ok += 1
        else:
            bad += 1
            print("MISMATCH", name, fmt, len(got), len(expected) - 11)
    print("ffmpeg agrees on", ok, "of", ok + bad)


def ffmpeg_fixtures():
    """PNGs written by ffmpeg's own encoder (its filter choices and zlib settings) from its testsrc2
    pattern; the expected samples are ffmpeg's own decode of them."""
    cases = [("ff_rgb24", "rgb24", "rgb24", 3, 8), ("ff_rgba64", "rgba64be", "rgba64le", 4, 16),
             ("ff_gray16", "gray16be", "gray16le", 1, 16), ("ff_pal8", "pal8", "rgb24", 3, 8),
             ("ff_monob", "monob", "gray", 1, 8), ("ff_ya8", "ya8", "ya8", 2, 8)]
    for name, enc_fmt, dec_fmt, ch, bits in cases:
        png = os.path.join(HERE, name + ".png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=36x22:rate=1",
                        "-frames:v", "1", "-pix_fmt", enc_fmt, png], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", png, "-f", "rawvideo", "-pix_fmt", dec_fmt, "-"],
                             capture_output=True, check=True).stdout
        with open(os.path.join(HERE, name + ".bin"), "wb") as f:
            f.write(struct.pack("<IIBBB", 36, 22, ch, bits, 0) + raw)
        print(name, len(raw))


if __name__ == "__main__":
    if "--check" in sys.argv:
        rest = sys.argv[sys.argv.index("--check") + 1:]
        check(rest[0] if rest else HERE)
    elif "--ffmpeg" in sys.argv:
        ffmpeg_fixtures()
    else:
        main()
