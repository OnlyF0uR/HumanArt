"""Makes the JPEG fixtures of humanart-io (dev-time only, [DEP-04]).

    python -I crates/io/tests/fixtures/jpeg/make_fixtures.py
    python -I crates/io/tests/fixtures/jpeg/make_fixtures.py --bench DIR   # large files for timing

Two kinds of fixture, every pixel from a fixed-seed pattern (no third-party content):

* `ff_*.jpg`: encoded by ffmpeg's MJPEG encoder (baseline, optimised Huffman tables) from a PPM the
  script writes. `ff_*.bin` is ffmpeg's own decoding of the file into its native planar format.
* `py_*.jpg`: encoded by the small JPEG writer below, independent of the Rust code: float FDCT, any
  sampling factors, sequential (interleaved or one scan per component) or progressive (the libjpeg
  "simple progression" scan script: spectral selection and successive approximation), restart
  intervals, optimal Huffman tables per scan (T.81 K.2), 8- and 16-bit quantisation tables, JFIF,
  Adobe APP14 (RGB, CMYK, YCCK), EXIF orientation. `py_*.bin` is the reference: for every component,
  a one-component baseline JPEG holding exactly that component's quantised coefficients is written
  and decoded by ffmpeg. Where ffmpeg also decodes the whole file natively (grey and YCbCr layouts),
  the script checks that its planes equal those references, which checks the writer's interleaving,
  progressive scans and restart markers against an independent decoder.

`<name>.bin` holds the component planes one after another, each `ceil(width * h / hmax)` by
`ceil(height * v / vmax)` samples, rows top to bottom: what the Rust decoder must produce before
upsampling and colour conversion (within the IDCT tolerance stated in the Rust tests).

`<name>_rgb.bin` (for the 4:4:4 files) is ffmpeg's conversion to RGB24.

`py_prog_*` files carry exactly the coefficients of their sequential twin named in TWINS, so the
Rust decoder must give identical planes for both.
"""

import math
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
          7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
          39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
LUMA_Q = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17,
          22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78,
          87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
CHROMA_Q = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66,
            99, 99] + [99] * 36


class Lcg:
    def __init__(self, seed):
        self.state = seed

    def below(self, n):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) % n


def scaled(table, quality):
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [min(max((q * scale + 50) // 100, 1), 255) for q in table]


def pattern(w, h, seed, channels=3):
    """Smooth gradients, a hard-edged disc and checkerboard, and noise: `channels` planes."""
    rng = Lcg(seed)
    planes = [[[0] * w for _ in range(h)] for _ in range(channels)]
    for y in range(h):
        for x in range(w):
            disc = (x - w * 0.6) ** 2 + (y - h * 0.4) ** 2 < (min(w, h) * 0.25) ** 2
            checker = x < w // 3 and y > h // 2 and ((x // 3 + y // 3) % 2 == 0)
            for c in range(channels):
                v = 128 + 90 * math.sin(x / (5.0 + 3 * c) + c) * math.cos(y / (7.0 + 2 * c) - c)
                if disc:
                    v = 230 - 60 * c
                if checker:
                    v = 20 if c % 2 else 240
                v += rng.below(17) - 8
                planes[c][y][x] = min(max(int(v), 0), 255)
    return planes


def rgb_to_ycc(planes):
    r, g, b = planes
    h, w = len(r), len(r[0])
    out = [[[0] * w for _ in range(h)] for _ in range(3)]
    for y in range(h):
        for x in range(w):
            R, G, B = r[y][x], g[y][x], b[y][x]
            vals = (0.299 * R + 0.587 * G + 0.114 * B,
                    128 - 0.168736 * R - 0.331264 * G + 0.5 * B,
                    128 + 0.5 * R - 0.418688 * G - 0.081312 * B)
            for c in range(3):
                out[c][y][x] = min(max(int(math.floor(vals[c] + 0.5)), 0), 255)
    return out


# --- Forward DCT and quantisation --------------------------------------------------------------------

DCT = [[(math.sqrt(0.5) if u == 0 else 1.0) / 2 * math.cos((2 * x + 1) * u * math.pi / 16) for x in range(8)]
       for u in range(8)]


def fdct_quant(block, quant):
    """block: 64 level-shifted samples (row-major). Returns 64 quantised coefficients, natural order."""
    tmp = [[sum(DCT[u][x] * block[y * 8 + x] for x in range(8)) for u in range(8)] for y in range(8)]
    out = [0] * 64
    for v in range(8):
        for u in range(8):
            f = sum(DCT[v][y] * tmp[y][u] for y in range(8))
            out[v * 8 + u] = int(math.floor(f / quant[v * 8 + u] + 0.5))
    return out


class Frame:
    """Sizes of a frame: comps are (id, h, v, quant table index)."""

    def __init__(self, w, h, comps):
        self.w, self.h, self.comps = w, h, comps
        self.hmax = max(c[1] for c in comps)
        self.vmax = max(c[2] for c in comps)
        self.mcux = -(-w // (8 * self.hmax))
        self.mcuy = -(-h // (8 * self.vmax))

    def size(self, i):
        _, h, v, _ = self.comps[i]
        return -(-self.w * h // self.hmax), -(-self.h * v // self.vmax)

    def grid(self, i):
        _, h, v, _ = self.comps[i]
        return self.mcux * h, self.mcuy * v


def coefficients(frame, planes, quants):
    """Downsamples (box filter) each full-resolution plane, pads by edge replication to the MCU grid,
    and returns per component a dict (bx, by) -> 64 quantised coefficients."""
    out = []
    for i, (_, h, v, tq) in enumerate(frame.comps):
        fx, fy = frame.hmax // h, frame.vmax // v
        cw, ch = frame.size(i)
        bw, bh = frame.grid(i)
        src = planes[i]
        comp = [[0] * cw for _ in range(ch)]
        for y in range(ch):
            for x in range(cw):
                total = 0
                for dy in range(fy):
                    for dx in range(fx):
                        total += src[min(y * fy + dy, frame.h - 1)][min(x * fx + dx, frame.w - 1)]
                comp[y][x] = (total + fx * fy // 2) // (fx * fy)
        blocks = {}
        for by in range(bh):
            for bx in range(bw):
                block = [comp[min(by * 8 + r, ch - 1)][min(bx * 8 + c, cw - 1)] - 128 for r in range(8) for c in range(8)]
                blocks[(bx, by)] = fdct_quant(block, quants[tq])
        out.append(blocks)
    return out


# --- Huffman coding ---------------------------------------------------------------------------------

def optimal_table(freq):
    """T.81 K.2 (libjpeg's jpeg_gen_optimal_table): (counts[16], symbols) for 256 frequencies."""
    freq = list(freq) + [1]
    codesize = [0] * 257
    others = [-1] * 257
    while True:
        c1, v = -1, None
        for i in range(257):
            if freq[i] and (v is None or freq[i] <= v):
                c1, v = i, freq[i]
        c2, v = -1, None
        for i in range(257):
            if freq[i] and i != c1 and (v is None or freq[i] <= v):
                c2, v = i, freq[i]
        if c2 < 0:
            break
        freq[c1] += freq[c2]
        freq[c2] = 0
        codesize[c1] += 1
        while others[c1] >= 0:
            c1 = others[c1]
            codesize[c1] += 1
        others[c1] = c2
        codesize[c2] += 1
        while others[c2] >= 0:
            c2 = others[c2]
            codesize[c2] += 1
    bits = [0] * 40
    for i in range(257):
        if codesize[i]:
            bits[codesize[i]] += 1
    for i in range(39, 16, -1):
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0:
                j -= 1
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
    i = 16
    while bits[i] == 0:
        i -= 1
    bits[i] -= 1
    symbols = [s for size in range(1, 40) for s in range(256) if codesize[s] == size]
    return bits[1:17], symbols


def canonical(counts, symbols):
    codes, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            codes[symbols[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return codes


class Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, v, n):
        for i in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((v >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc, self.n = 0, 0

    def flush(self):
        while self.n:
            self.put(1, 1)


class Coder:
    """Counts symbols (first pass) or writes them (second pass) for the tables of one scan."""

    def __init__(self, tables=None):
        self.tables = tables
        self.freq = {}
        self.bits = Bits()

    def sym(self, table, s):
        if self.tables is None:
            self.freq.setdefault(table, [0] * 256)[s] += 1
        else:
            code, length = self.tables[table][s]
            self.bits.put(code, length)

    def raw(self, v, n):
        if self.tables is not None and n:
            self.bits.put(v & ((1 << n) - 1), n)


def nbits(v):
    v = abs(v)
    n = 0
    while v:
        n += 1
        v >>= 1
    return n


def put_value(coder, v, n):
    coder.raw(v if v >= 0 else v - 1, n)


class ScanState:
    def __init__(self, coder):
        self.coder = coder
        self.eobrun = 0
        self.corrections = []  # buffered refinement bits

    def flush_eobrun(self, table):
        if self.eobrun:
            n = nbits(self.eobrun) - 1
            self.coder.sym(table, n << 4)
            self.coder.raw(self.eobrun, n)
            self.eobrun = 0
        for b in self.corrections:
            self.coder.raw(b, 1)
        self.corrections = []


def encode_block_sequential(st, coef, pred, dc_t, ac_t):
    diff = coef[0] - pred
    n = nbits(diff)
    st.coder.sym(dc_t, n)
    put_value(st.coder, diff, n)
    run = 0
    for k in range(1, 64):
        v = coef[ZIGZAG[k]]
        if v == 0:
            run += 1
            continue
        while run > 15:
            st.coder.sym(ac_t, 0xF0)
            run -= 16
        n = nbits(v)
        st.coder.sym(ac_t, (run << 4) | n)
        put_value(st.coder, v, n)
        run = 0
    if run:
        st.coder.sym(ac_t, 0)
    return coef[0]


def shifted(v, al):
    """The point transform of T.81 G.1.2.1 for AC values: the magnitude shifted, sign kept."""
    return (abs(v) >> al) * (1 if v >= 0 else -1)


def encode_ac_first(st, coef, ss, se, al, t):
    run = 0
    for k in range(ss, se + 1):
        v = shifted(coef[ZIGZAG[k]], al)
        if v == 0:
            run += 1
            continue
        st.flush_eobrun(t)
        while run > 15:
            st.coder.sym(t, 0xF0)
            run -= 16
        n = nbits(v)
        st.coder.sym(t, (run << 4) | n)
        put_value(st.coder, v, n)
        run = 0
    if run:
        st.eobrun += 1
        if st.eobrun == 0x7FFF:
            st.flush_eobrun(t)


def encode_ac_refine(st, coef, ss, se, al, t):
    absv = [abs(coef[ZIGZAG[k]]) >> al for k in range(64)]
    eob = 0
    for k in range(ss, se + 1):
        if absv[k] == 1:
            eob = k
    run = 0
    pending = []
    for k in range(ss, se + 1):
        a = absv[k]
        if a == 0:
            run += 1
            continue
        while run > 15 and k <= eob:
            st.flush_eobrun(t)
            st.coder.sym(t, 0xF0)
            run -= 16
            for b in pending:
                st.coder.raw(b, 1)
            pending = []
        if a > 1:
            pending.append(a & 1)
            continue
        st.flush_eobrun(t)
        st.coder.sym(t, (run << 4) | 1)
        st.coder.raw(1 if coef[ZIGZAG[k]] > 0 else 0, 1)
        for b in pending:
            st.coder.raw(b, 1)
        pending = []
        run = 0
    if run or pending:
        st.eobrun += 1
        st.corrections += pending
        if st.eobrun == 0x7FFF or len(st.corrections) > 900:
            st.flush_eobrun(t)


def scan_mcus(frame, members):
    """The blocks of each MCU of a scan over component indices `members`."""
    if len(members) == 1:
        i = members[0]
        cw, ch = frame.size(i)
        for by in range(-(-ch // 8)):
            for bx in range(-(-cw // 8)):
                yield [(i, bx, by)]
    else:
        for my in range(frame.mcuy):
            for mx in range(frame.mcux):
                yield [(i, mx * frame.comps[i][1] + x, my * frame.comps[i][2] + y)
                       for i in members for y in range(frame.comps[i][2]) for x in range(frame.comps[i][1])]


def code_scan(coder, frame, coefs, scan, restart):
    members, ss, se, ah, al = scan
    st = ScanState(coder)
    preds = {i: 0 for i in members}
    for m, mcu in enumerate(scan_mcus(frame, members)):
        if restart and m and m % restart == 0:
            st.flush_eobrun(("ac", 0))
            if coder.tables is not None:
                coder.bits.flush()
                coder.bits.out += bytes([0xFF, 0xD0 + (m // restart - 1) % 8])
            preds = {i: 0 for i in members}
        for (i, bx, by) in mcu:
            coef = coefs[i][(bx, by)]
            slot = members.index(i)
            if ss == 0 and se == 63:
                preds[i] = encode_block_sequential(st, coef, preds[i], ("dc", slot), ("ac", slot))
            elif ss == 0 and ah == 0:
                dc = coef[0] >> al
                n = nbits(dc - preds[i])
                coder.sym(("dc", slot), n)
                put_value(coder, dc - preds[i], n)
                preds[i] = dc
            elif ss == 0:
                coder.raw((coef[0] >> al) & 1, 1)
            elif ah == 0:
                encode_ac_first(st, coef, ss, se, al, ("ac", 0))
            else:
                encode_ac_refine(st, coef, ss, se, al, ("ac", 0))
    st.flush_eobrun(("ac", 0))
    coder.bits.flush()


def segment(marker, payload):
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


SIMPLE_PROGRESSION_3 = [  # libjpeg jcparam.c jpeg_simple_progression for YCbCr
    ([0, 1, 2], 0, 0, 0, 1), ([0], 1, 5, 0, 2), ([2], 1, 63, 0, 1), ([1], 1, 63, 0, 1), ([0], 6, 63, 0, 2),
    ([0], 1, 63, 2, 1), ([0, 1, 2], 0, 0, 1, 0), ([2], 1, 63, 1, 0), ([1], 1, 63, 1, 0), ([0], 1, 63, 1, 0)]
SIMPLE_PROGRESSION_1 = [([0], 0, 0, 0, 1), ([0], 1, 5, 0, 2), ([0], 6, 63, 0, 2), ([0], 1, 63, 2, 1),
                        ([0], 0, 0, 1, 0), ([0], 1, 63, 1, 0)]


def write_jpeg(frame, coefs, quants, scans, restart=0, progressive=False, jfif=True, adobe=None, exif=None,
               wide_quant=False):
    out = bytearray(b"\xFF\xD8")
    if jfif:
        out += segment(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0")
    if exif is not None:
        out += segment(0xE1, exif)
    if adobe is not None:
        out += segment(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    for t, q in enumerate(quants):
        if wide_quant:
            out += segment(0xDB, bytes([0x10 | t]) + b"".join(struct.pack(">H", q[z]) for z in ZIGZAG))
        else:
            out += segment(0xDB, bytes([t]) + bytes(q[z] for z in ZIGZAG))
    sof = struct.pack(">BHHB", 8, frame.h, frame.w, len(frame.comps))
    for (cid, h, v, tq) in frame.comps:
        sof += bytes([cid, (h << 4) | v, tq])
    out += segment(0xC2 if progressive else 0xC0, sof)
    if restart:
        out += segment(0xDD, struct.pack(">H", restart))
    for scan in scans:
        counter = Coder()
        code_scan(counter, frame, coefs, scan, restart)
        tables, dht = {}, b""
        for (kind, slot), freq in sorted(counter.freq.items()):
            counts, symbols = optimal_table(freq)
            tables[(kind, slot)] = canonical(counts, symbols)
            dht += bytes([(0 if kind == "dc" else 0x10) | slot]) + bytes(counts) + bytes(symbols)
        if dht:
            out += segment(0xC4, dht)
        members, ss, se, ah, al = scan
        sos = bytes([len(members)])
        for slot, i in enumerate(members):
            sos += bytes([frame.comps[i][0], (slot << 4) | (0 if ss else slot)])
        out += segment(0xDA, sos + bytes([ss, se, (ah << 4) | al]))
        writer = Coder(tables)
        code_scan(writer, frame, coefs, scan, restart)
        out += writer.bits.out
    return bytes(out + b"\xFF\xD9")


def exif_block(orientation, big_endian):
    e = ">" if big_endian else "<"
    tiff = (b"MM\0\x2A" if big_endian else b"II\x2A\0") + struct.pack(e + "I", 8)
    tiff += struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0x0112, 3, 1) + struct.pack(e + "HH", orientation, 0)
    tiff += struct.pack(e + "I", 0)
    return b"Exif\0\0" + tiff


# --- ffmpeg ------------------------------------------------------------------------------------------

def ffmpeg_planes(path, pix_fmt):
    return subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-i", path, "-f", "rawvideo", "-pix_fmt",
                           pix_fmt, "-"], check=True, capture_output=True).stdout


def ffmpeg_rgb(name):
    """`<name>_rgb.bin`: ffmpeg's RGB conversion with accurate rounding, a cross-check of the Rust
    colour conversion. Only for 4:4:4 files: swscale places and filters subsampled chroma its own
    way (none of nearest, centred or co-sited bilinear reproduces it), so it is no reference for
    JFIF's centred upsampling."""
    path = os.path.join(HERE, name + ".jpg")
    rgb = subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-i", path, "-sws_flags",
                          "accurate_rnd+full_chroma_int+bitexact", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    with open(os.path.join(HERE, name + "_rgb.bin"), "wb") as f:
        f.write(rgb)


def ffmpeg_pix_fmt(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return out.strip()


def reference_planes(name, frame, coefs, quants):
    """ffmpeg's decoding of each component, through a one-component twin holding its coefficients."""
    out = b""
    for i, (_, _, _, tq) in enumerate(frame.comps):
        cw, ch = frame.size(i)
        twin = Frame(cw, ch, [(1, 1, 1, 0)])
        data = write_jpeg(twin, [coefs[i]], [quants[tq]], [([0], 0, 63, 0, 0)])
        path = os.path.join(HERE, "_twin.jpg")
        with open(path, "wb") as f:
            f.write(data)
        plane = ffmpeg_planes(path, "gray")
        os.remove(path)
        assert len(plane) == cw * ch, (name, i, len(plane))
        out += plane
    return out


NATIVE = {((1, 1),): "gray", ((1, 1), (1, 1), (1, 1)): "yuvj444p", ((2, 2), (1, 1), (1, 1)): "yuvj420p",
          ((2, 1), (1, 1), (1, 1)): "yuvj422p", ((1, 2), (1, 1), (1, 1)): "yuvj440p",
          ((4, 1), (1, 1), (1, 1)): "yuvj411p"}


def py_fixture(name, w, h, sampling, quality=80, mode="interleaved", restart=0, kind="ycc", seed=1, exif=None,
               wide_quant=False, quants=None, twin_of=None):
    channels = 1 if kind == "grey" else 4 if kind in ("cmyk", "ycck") else 3
    src = pattern(w, h, seed, max(channels, 3))
    if kind == "grey":
        planes = [rgb_to_ycc(src[:3])[0]]
    elif kind in ("ycc", "ycck"):
        planes = rgb_to_ycc(src[:3]) + ([src[3]] if kind == "ycck" else [])
    else:
        planes = src[:channels]
    ids = [ord(c) for c in "RGB"] if kind == "rgb" else list(range(1, channels + 1))
    comps = [(ids[i], sampling[i][0], sampling[i][1], 0 if i in (0, 3) else 1) for i in range(channels)]
    frame = Frame(w, h, comps)
    if quants is None:
        quants = [scaled(LUMA_Q, quality), scaled(CHROMA_Q, quality)][:1 if channels == 1 else 2]
    coefs = coefficients(frame, planes, quants)
    members = list(range(channels))
    if mode == "progressive":
        scans = SIMPLE_PROGRESSION_1 if channels == 1 else SIMPLE_PROGRESSION_3
    elif mode == "separate":
        scans = [([i], 0, 63, 0, 0) for i in members]
    else:
        scans = [(members, 0, 63, 0, 0)]
    adobe = {"rgb": 0, "cmyk": 0, "ycck": 2}.get(kind)
    data = write_jpeg(frame, coefs, quants, scans, restart, mode == "progressive", jfif=adobe is None, adobe=adobe,
                      exif=exif, wide_quant=wide_quant)
    path = os.path.join(HERE, name + ".jpg")
    with open(path, "wb") as f:
        f.write(data)
    ref = reference_planes(name, frame, coefs, quants)
    native = NATIVE.get(tuple(tuple(s) for s in sampling[:channels])) if kind in ("ycc", "grey") else None
    if native:
        got = ffmpeg_planes(path, native)
        assert got == ref, f"{name}: ffmpeg's native decoding differs from the per-component references"
    with open(os.path.join(HERE, name + ".bin"), "wb") as f:
        f.write(ref)
    print(f"{name}: {len(data)} bytes, {'native check' if native else 'twins only'}")
    return coefs


def ff_fixture(name, w, h, pix_fmt, q):
    src = pattern(w, h, 7)
    ppm = os.path.join(HERE, "_src.ppm")
    with open(ppm, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (w, h))
        f.write(bytes(src[c][y][x] for y in range(h) for x in range(w) for c in range(3)))
    path = os.path.join(HERE, name + ".jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ppm, "-pix_fmt", pix_fmt, "-q:v", str(q), "-frames:v", "1",
                    path], check=True)
    os.remove(ppm)
    with open(os.path.join(HERE, name + ".bin"), "wb") as f:
        f.write(ffmpeg_planes(path, pix_fmt))
    print(f"{name}: {os.path.getsize(path)} bytes, ffmpeg-encoded")


def bench(folder):
    """A 1024x768 4:2:0 image as progressive and as sequential JPEG, for the ignored Rust timing test
    (`HUMANART_JPEG_BENCH=DIR`). Not fixtures: they stay out of the tree."""
    w, h = 1024, 768
    planes = rgb_to_ycc(pattern(w, h, 21))
    frame = Frame(w, h, [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)])
    quants = [scaled(LUMA_Q, 90), scaled(CHROMA_Q, 90)]
    coefs = coefficients(frame, planes, quants)
    for name, scans, progressive in (("bench_prog", SIMPLE_PROGRESSION_3, True),
                                     ("bench_seq", [([0, 1, 2], 0, 63, 0, 0)], False)):
        data = write_jpeg(frame, coefs, quants, scans, progressive=progressive)
        with open(os.path.join(folder, name + ".jpg"), "wb") as f:
            f.write(data)
        print(f"{name}: {len(data)} bytes")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--bench":
        return bench(sys.argv[2])
    s444, s420, s422 = [(1, 1)] * 3, [(2, 2), (1, 1), (1, 1)], [(2, 1), (1, 1), (1, 1)]
    py_fixture("py_grey", 37, 29, [(1, 1)], kind="grey", seed=2)
    py_fixture("py_444", 33, 17, s444, quality=92, seed=3, wide_quant=True)
    py_fixture("py_420", 45, 31, s420, seed=4)
    py_fixture("py_420_restart", 45, 31, s420, restart=3, seed=4)
    py_fixture("py_422_separate", 41, 27, s422, mode="separate", seed=5)
    py_fixture("py_440", 40, 36, [(1, 2), (1, 1), (1, 1)], seed=6)
    py_fixture("py_411", 50, 20, [(4, 1), (1, 1), (1, 1)], seed=7)
    py_fixture("py_odd", 39, 35, [(2, 2), (1, 2), (2, 1)], seed=8)
    py_fixture("py_chroma_full", 26, 22, [(1, 1), (2, 2), (1, 1)], seed=9)
    py_fixture("py_prog_420", 45, 31, s420, mode="progressive", seed=4)
    py_fixture("py_prog_420_restart", 45, 31, s420, mode="progressive", restart=2, seed=4)
    py_fixture("py_prog_grey", 37, 29, [(1, 1)], kind="grey", mode="progressive", seed=2)
    py_fixture("py_prog_444", 33, 17, s444, quality=92, mode="progressive", seed=3, wide_quant=True)
    py_fixture("py_rgb", 24, 20, s444, kind="rgb", seed=10)
    py_fixture("py_cmyk", 24, 20, [(1, 1)] * 4, kind="cmyk", seed=11)
    py_fixture("py_ycck", 32, 24, [(2, 2), (1, 1), (1, 1), (2, 2)], kind="ycck", seed=12)
    py_fixture("py_exif6", 20, 12, s420, seed=13, exif=exif_block(6, True))
    py_fixture("py_exif8", 20, 12, s420, seed=13, exif=exif_block(8, False))
    py_fixture("py_big", 160, 120, s420, quality=90, seed=14)
    ff_fixture("ff_420", 61, 47, "yuvj420p", 3)
    ff_fixture("ff_422", 61, 47, "yuvj422p", 2)
    ff_fixture("ff_444", 61, 47, "yuvj444p", 5)
    for name in ("ff_444", "py_444"):
        ffmpeg_rgb(name)


if __name__ == "__main__":
    sys.exit(main())
