"""Dev-time cross-check [DEP-04]: ffmpeg decodes every EXR our encoder wrote (test
`write_cross_check_files`) and must get exactly the samples our decoder got.

usage: python xcheck.py <dir>
Files: c<channels>_<compression>_<half|float>_<w>x<h>.exr and .ours.f32 (interleaved f32 LE,
R G B A or Y). ffmpeg is asked for its decoder's native format (planar G B R [A] / Y, f16 or f32).
"""
import os
import re
import struct
import subprocess
import sys


def main():
    d = sys.argv[1]
    ok = bad = 0
    for name in sorted(os.listdir(d)):
        m = re.match(r"c(\d)_(\w+?)_(half|float)_(\d+)x(\d+)\.exr$", name)
        if not m:
            continue
        ch, comp, ty, w, h = int(m[1]), m[2], m[3], int(m[4]), int(m[5])
        base = {1: "gray", 3: "gbrp", 4: "gbrap"}[ch]
        fmt = base + ("f16le" if ty == "half" else "f32le")
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", os.path.join(d, name),
             "-f", "rawvideo", "-c:v", "rawvideo", "-pix_fmt", fmt, "-"],
            capture_output=True)
        n = w * h
        size = 2 if ty == "half" else 4
        if out.returncode != 0 or len(out.stdout) != n * ch * size:
            print("FAIL decode", name, out.stderr.decode()[-200:])
            bad += 1
            continue
        code = "<%d%s" % (n, "e" if ty == "half" else "f")
        planes = [struct.unpack_from(code, out.stdout, k * n * size) for k in range(ch)]
        # ffmpeg plane order -> our interleaved order (R G B A or Y).
        order = {1: [0], 3: [2, 0, 1], 4: [2, 0, 1, 3]}[ch]
        ours = struct.unpack("<%df" % (n * ch), open(os.path.join(d, name[:-4] + ".ours.f32"), "rb").read())
        diff = sum(1 for p in range(n) for c in range(ch)
                   if struct.pack("<f", planes[order[c]][p]) != struct.pack("<f", ours[p * ch + c]))
        if diff:
            print("FAIL", name, diff, "samples differ")
            bad += 1
        else:
            ok += 1
    print("ok", ok, "failed", bad)


main()
