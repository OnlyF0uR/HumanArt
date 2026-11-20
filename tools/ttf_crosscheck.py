#!/usr/bin/env python3
"""Independent reading of TrueType facts, for cross-checking crates/io/src/ttf.

Uses only the standard library (`struct`). It shares no code with the Rust parser: the expected
numbers in the Rust tests that are marked "cross-checked with tools/ttf_crosscheck.py" were produced
by running

    python -I tools/ttf_crosscheck.py <font.ttf> [more fonts...]

and copying what it printed. Kerning is read from GPOS (feature `kern` under DFLT/latn, lookup type 2
formats 1 and 2, extension lookups unwrapped) and from `kern` format 0, each reported separately.

Besides the facts about single glyphs it prints whole-font aggregates (every cmap mapping, every
advance, every outline with composites assembled, the kerning of every pair of printable ASCII
characters), so that the Rust tests can compare all glyphs and not only a sample.

The committed fixtures the Rust tests compare against line by line are written with

    python -I tools/ttf_crosscheck.py --fixtures crates/io/tests/fixtures/ttf data/fonts/*.ttf

one text file per font (see `write_fixture` for the line format): the font-wide metrics, every cmap
mapping, for every glyph its metrics, stored box, outline size, segment counts, control-point bounds
and exact (curve) bounds, and every non-zero kerning among the glyphs of U+0020..U+007E and
U+00A0..U+00FF.
"""
import struct
import sys


def u16(d, o):
    return struct.unpack_from(">H", d, o)[0]


def i16(d, o):
    return struct.unpack_from(">h", d, o)[0]


def u32(d, o):
    return struct.unpack_from(">I", d, o)[0]


def tables(d):
    n = u16(d, 4)
    out = {}
    for i in range(n):
        tag, _cs, off, ln = struct.unpack_from(">4sIII", d, 12 + 16 * i)
        out[tag.decode("latin-1")] = d[off:off + ln]
    return out


def cmap_lookup(cmap, cp):
    """Glyph id of a code point: the (3,10) format 12 subtable if present, else (3,1) format 4."""
    n = u16(cmap, 2)
    recs = {}
    for i in range(n):
        pid, eid, off = struct.unpack_from(">HHI", cmap, 4 + 8 * i)
        recs[(pid, eid)] = off
    for key in ((3, 10), (0, 4), (0, 6), (3, 1), (0, 3)):
        if key not in recs:
            continue
        o = recs[key]
        fmt = u16(cmap, o)
        if fmt == 12:
            ngroups = u32(cmap, o + 12)
            for g in range(ngroups):
                s, e, gid = struct.unpack_from(">III", cmap, o + 16 + 12 * g)
                if s <= cp <= e:
                    return gid + (cp - s)
            return 0
        if fmt == 4:
            if cp > 0xFFFF:
                return 0
            segx2 = u16(cmap, o + 6)
            seg = segx2 // 2
            ends = o + 14
            starts = ends + segx2 + 2
            deltas = starts + segx2
            ranges = deltas + segx2
            for s in range(seg):
                if cp <= u16(cmap, ends + 2 * s):
                    start = u16(cmap, starts + 2 * s)
                    if cp < start:
                        return 0
                    delta = u16(cmap, deltas + 2 * s)
                    ro = u16(cmap, ranges + 2 * s)
                    if ro == 0:
                        return (cp + delta) & 0xFFFF
                    g = u16(cmap, ranges + 2 * s + ro + 2 * (cp - start))
                    return (g + delta) & 0xFFFF if g else 0
            return 0
    raise SystemExit("no usable cmap subtable")


def cmap_formats(cmap):
    n = u16(cmap, 2)
    out = []
    for i in range(n):
        pid, eid, off = struct.unpack_from(">HHI", cmap, 4 + 8 * i)
        out.append((pid, eid, u16(cmap, off)))
    return out


def advance(t, gid):
    nhm = u16(t["hhea"], 34)
    g = min(gid, nhm - 1)
    return u16(t["hmtx"], 4 * g)


def lsb(t, gid):
    nhm = u16(t["hhea"], 34)
    if gid < nhm:
        return i16(t["hmtx"], 4 * gid + 2)
    return i16(t["hmtx"], 4 * nhm + 2 * (gid - nhm))


def glyph_range(t, gid):
    long = i16(t["head"], 50)
    if long:
        return u32(t["loca"], 4 * gid), u32(t["loca"], 4 * gid + 4)
    return 2 * u16(t["loca"], 2 * gid), 2 * u16(t["loca"], 2 * gid + 2)


def glyph_header(t, gid):
    a, b = glyph_range(t, gid)
    if a == b:
        return None
    return struct.unpack_from(">hhhhh", t["glyf"], a)


def simple_point_count(t, gid):
    a, _b = glyph_range(t, gid)
    n = i16(t["glyf"], a)
    if n <= 0:
        return None
    return u16(t["glyf"], a + 10 + 2 * (n - 1)) + 1


def table_checksum(b):
    b = b + b"\0" * (-len(b) % 4)
    return sum(struct.unpack(">%dI" % (len(b) // 4), b)) & 0xFFFFFFFF


def checksums(d):
    """(number of tables, tables whose checksum is wrong, checksum of the whole file)."""
    n = u16(d, 4)
    bad = []
    for i in range(n):
        tag, cs, off, ln = struct.unpack_from(">4sIII", d, 12 + 16 * i)
        b = d[off:off + ln]
        if tag == b"head":
            b = b[:8] + b"\0\0\0\0" + b[12:]
        if table_checksum(b) != cs:
            bad.append(tag.decode("latin-1"))
    return n, bad, table_checksum(d)


def cmap_all(cmap, num_glyphs):
    """Every (code point, glyph) of the preferred Unicode subtable, glyph 0 and out-of-range ids dropped."""
    n = u16(cmap, 2)
    recs = {}
    for i in range(n):
        pid, eid, off = struct.unpack_from(">HHI", cmap, 4 + 8 * i)
        recs[(pid, eid)] = off
    out = {}
    for key in ((3, 10), (0, 4), (0, 6), (3, 1), (0, 3)):
        if key not in recs:
            continue
        o = recs[key]
        fmt = u16(cmap, o)
        if fmt == 12:
            for g in range(u32(cmap, o + 12)):
                s, e, gid = struct.unpack_from(">III", cmap, o + 16 + 12 * g)
                for cp in range(s, e + 1):
                    out[cp] = gid + cp - s
        elif fmt == 4:
            segx2 = u16(cmap, o + 6)
            ends = o + 14
            starts = ends + segx2 + 2
            deltas = starts + segx2
            ranges = deltas + segx2
            for sgm in range(segx2 // 2):
                end, start = u16(cmap, ends + 2 * sgm), u16(cmap, starts + 2 * sgm)
                delta, ro = u16(cmap, deltas + 2 * sgm), u16(cmap, ranges + 2 * sgm)
                for cp in range(start, end + 1):
                    if ro == 0:
                        out[cp] = (cp + delta) & 0xFFFF
                    else:
                        g = u16(cmap, ranges + 2 * sgm + ro + 2 * (cp - start))
                        out[cp] = (g + delta) & 0xFFFF if g else 0
        else:
            continue
        break
    # Surrogates are not characters.
    return {cp: g for cp, g in out.items() if 0 < g < num_glyphs and not 0xD800 <= cp <= 0xDFFF and cp <= 0x10FFFF}


def simple_glyph(g, a, n):
    """Points [x, y, on_curve] and exclusive contour ends of the simple glyph at offset a with n contours."""
    ends = [u16(g, a + 10 + 2 * i) + 1 for i in range(n)]
    npts = ends[-1] if ends else 0
    o = a + 10 + 2 * n
    o += 2 + u16(g, o)
    flags = []
    while len(flags) < npts:
        f = g[o]
        o += 1
        flags.append(f)
        if f & 8:
            flags.extend([f] * g[o])
            o += 1
    flags = flags[:npts]
    coords = []
    for short, same in ((2, 16), (4, 32)):
        v = 0
        axis = []
        for f in flags:
            if f & short:
                v += g[o] if f & same else -g[o]
                o += 1
            elif not f & same:
                v += i16(g, o)
                o += 2
            axis.append(v)
        coords.append(axis)
    return [[coords[0][i], coords[1][i], flags[i] & 1] for i in range(npts)], ends


def components(g, a):
    """The component records of the composite glyph at offset a: (flags, gid, arg1, arg2, a, b, c, d)."""
    o = a + 10
    out = []
    while True:
        flags, gid = u16(g, o), u16(g, o + 2)
        o += 4
        if flags & 1:
            a1, a2 = (i16(g, o), i16(g, o + 2)) if flags & 2 else (u16(g, o), u16(g, o + 2))
            o += 4
        else:
            a1, a2 = struct.unpack_from(">bb" if flags & 2 else ">BB", g, o)
            o += 2
        xs, s01, s10, ys = 1.0, 0.0, 0.0, 1.0
        if flags & 0x8:
            xs = ys = i16(g, o) / 16384.0
            o += 2
        elif flags & 0x40:
            xs, ys = i16(g, o) / 16384.0, i16(g, o + 2) / 16384.0
            o += 4
        elif flags & 0x80:
            xs, s01, s10, ys = (i16(g, o + 2 * k) / 16384.0 for k in range(4))
            o += 8
        out.append((flags, gid, a1, a2, xs, s01, s10, ys))
        if not flags & 0x20:
            return out


def assemble(t, gid, stats=None):
    """Points, exclusive contour ends and total component count of a glyph, composites assembled."""
    a, b = glyph_range(t, gid)
    if a == b:
        return [], [], 0
    g = t["glyf"]
    n = i16(g, a)
    if n >= 0:
        pts, ends = simple_glyph(g, a, n)
        return pts, ends, 0
    pts, ends, count = [], [], 0
    for flags, cg, a1, a2, xs, s01, s10, ys in components(g, a):
        cp, ce, cc = assemble(t, cg)
        count += 1 + cc
        cp = [[xs * x + s10 * y, s01 * x + ys * y, on] for x, y, on in cp]
        if flags & 2:
            dx, dy = float(a1), float(a2)
            if flags & 0x800 and not flags & 0x1000:
                dx *= (xs * xs + s10 * s10) ** 0.5
                dy *= (ys * ys + s01 * s01) ** 0.5
        else:
            dx, dy = pts[a1][0] - cp[a2][0], pts[a1][1] - cp[a2][1]
        if stats is not None:
            for name, bit in (("scale", 0x8), ("xy_scale", 0x40), ("two_by_two", 0x80), ("use_my_metrics", 0x200),
                              ("scaled_offset", 0x800)):
                if flags & bit:
                    stats.setdefault(name, []).append(gid)
            if not flags & 2:
                stats.setdefault("point_matching", []).append(gid)
            if i16(g, glyph_range(t, cg)[0]) < 0 if glyph_range(t, cg)[0] != glyph_range(t, cg)[1] else False:
                stats.setdefault("nested", []).append(gid)
        ends.extend(len(pts) + e for e in ce)
        pts.extend([x + dx, y + dy, on] for x, y, on in cp)
    return pts, ends, count


def segments(pts, ends):
    """(lines, quads) of the contours as path segments, the closing straight line not counted.

    Every off-curve point is the control point of exactly one quadratic. Every pair of consecutive
    on-curve points is one line, except the pair that ends at the contour's starting point (the first
    point, or the last one when the first is off-curve), which the path leaves to its close command.
    Contours of fewer than two points draw nothing.
    """
    lines = quads = 0
    first = 0
    for end in ends:
        c = pts[first:end]
        first = end
        if len(c) < 2:
            continue
        quads += sum(1 for p in c if not p[2])
        n = len(c)
        on_on = sum(1 for i in range(n) if c[i][2] and c[(i + 1) % n][2])
        if c[0][2]:
            start = 0
        elif c[-1][2]:
            start = n - 1
        else:
            start = None
        if start is not None and c[(start - 1) % n][2]:
            on_on -= 1
        lines += on_on
    return lines, quads


def drawn_bounds(pts, ends):
    """Min/max of the points of every contour that draws something, or None."""
    xs, ys = [], []
    first = 0
    for end in ends:
        if end - first >= 2:
            xs.extend(p[0] for p in pts[first:end])
            ys.extend(p[1] for p in pts[first:end])
        first = end
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def metrics_source(t, gid, depth=0):
    """The glyph whose hmtx entry applies, following USE_MY_METRICS (last flagged component)."""
    a, b = glyph_range(t, gid)
    if a == b or i16(t["glyf"], a) >= 0 or depth >= 8:
        return gid
    flagged = [c[1] for c in components(t["glyf"], a) if c[0] & 0x200]
    return metrics_source(t, flagged[-1], depth + 1) if flagged else gid


def coverage_index(d, o, gid):
    fmt = u16(d, o)
    if fmt == 1:
        for i in range(u16(d, o + 2)):
            if u16(d, o + 4 + 2 * i) == gid:
                return i
        return None
    for i in range(u16(d, o + 2)):
        s, e, start = struct.unpack_from(">HHH", d, o + 4 + 6 * i)
        if s <= gid <= e:
            return start + gid - s
    return None


def class_of(d, o, gid):
    fmt = u16(d, o)
    if fmt == 1:
        start, count = u16(d, o + 2), u16(d, o + 4)
        if start <= gid < start + count:
            return u16(d, o + 6 + 2 * (gid - start))
        return 0
    for i in range(u16(d, o + 2)):
        s, e, c = struct.unpack_from(">HHH", d, o + 4 + 6 * i)
        if s <= gid <= e:
            return c
    return 0


def value_size(fmt):
    return 2 * bin(fmt).count("1")


def x_advance(d, o, fmt):
    if not fmt & 4:
        return 0
    return i16(d, o + 2 * bin(fmt & 3).count("1"))


def pair_subtable(d, o, left, right):
    """(applies, x_advance) of one PairPos subtable."""
    fmt = u16(d, o)
    cov = coverage_index(d, o + u16(d, o + 2), left)
    if cov is None:
        return False, 0
    vf1, vf2 = u16(d, o + 4), u16(d, o + 6)
    if fmt == 1:
        ps = o + u16(d, o + 10 + 2 * cov)
        rec = 2 + value_size(vf1) + value_size(vf2)
        for i in range(u16(d, ps)):
            if u16(d, ps + 2 + rec * i) == right:
                return True, x_advance(d, ps + 2 + rec * i + 2, vf1)
        return False, 0
    if fmt == 2:
        c1 = class_of(d, o + u16(d, o + 8), left)
        c2 = class_of(d, o + u16(d, o + 10), right)
        n1, n2 = u16(d, o + 12), u16(d, o + 14)
        if c1 >= n1 or c2 >= n2:
            return False, 0
        rec = value_size(vf1) + value_size(vf2)
        return True, x_advance(d, o + 16 + rec * (c1 * n2 + c2), vf1)
    return False, 0


def gpos_kern_lookups(g):
    script_list, feature_list = u16(g, 4), u16(g, 6)
    feats = set()
    found_script = False
    for i in range(u16(g, script_list)):
        tag = g[script_list + 2 + 6 * i:script_list + 6 + 6 * i]
        if tag not in (b"DFLT", b"latn"):
            continue
        found_script = True
        script = script_list + u16(g, script_list + 6 + 6 * i)
        dflt = u16(g, script)
        if dflt:
            ls = script + dflt
            if u16(g, ls + 2) != 0xFFFF:
                feats.add(u16(g, ls + 2))
            for k in range(u16(g, ls + 4)):
                feats.add(u16(g, ls + 6 + 2 * k))
    if not found_script:
        feats = set(range(u16(g, feature_list)))
    lookups = set()
    for f in feats:
        tag = g[feature_list + 2 + 6 * f:feature_list + 6 + 6 * f]
        if tag != b"kern":
            continue
        ft = feature_list + u16(g, feature_list + 6 + 6 * f)
        for k in range(u16(g, ft + 2)):
            lookups.add(u16(g, ft + 4 + 2 * k))
    return sorted(lookups)


def gpos_kern(t, left, right):
    """Sum of XAdvance of the first glyph over the kern feature's pair lookups, or None without GPOS kerning."""
    if "GPOS" not in t:
        return None
    g = t["GPOS"]
    lookups = gpos_kern_lookups(g)
    lookup_list = u16(g, 8)
    total = 0
    pair_lookups = 0
    for li in lookups:
        lk = lookup_list + u16(g, lookup_list + 2 + 2 * li)
        ltype = u16(g, lk)
        any_pair = False
        applied = False
        for s in range(u16(g, lk + 4)):
            st = lk + u16(g, lk + 6 + 2 * s)
            styp = ltype
            if styp == 9:
                styp = u16(g, st + 2)
                st = st + u32(g, st + 4)
            if styp != 2:
                continue
            any_pair = True
            if applied:
                continue
            ok, v = pair_subtable(g, st, left, right)
            if ok:
                total += v
                applied = True
        pair_lookups += any_pair
    return total if pair_lookups else None


def kern_table(t, left, right):
    if "kern" not in t:
        return None
    k = t["kern"]
    if u16(k, 0) != 0:
        return None
    o = 4
    total = 0
    for _ in range(u16(k, 2)):
        length, cov = u16(k, o + 2), u16(k, o + 4)
        if cov >> 8 == 0 and cov & 1 and not cov & 6:
            for i in range(u16(k, o + 6)):
                a, b, v = struct.unpack_from(">HHh", k, o + 14 + 6 * i)
                if (a, b) == (left, right):
                    total += v
        o += length
    return total


def name_string(t, name_id):
    n = t["name"]
    count, strings = u16(n, 2), u16(n, 4)
    for i in range(count):
        pid, eid, lang, nid, ln, off = struct.unpack_from(">HHHHHH", n, 6 + 12 * i)
        if nid == name_id and pid == 3 and lang == 0x409:
            return n[strings + off:strings + off + ln].decode("utf-16-be")
    for i in range(count):
        pid, eid, lang, nid, ln, off = struct.unpack_from(">HHHHHH", n, 6 + 12 * i)
        if nid == name_id and pid == 1:
            return n[strings + off:strings + off + ln].decode("mac_roman")
    return None


def report(path, pairs, chars):
    d = open(path, "rb").read()
    t = tables(d)
    print("==", path.replace("\\", "/").split("/")[-1], len(d), "bytes")
    print("sfnt version %08x" % u32(d, 0), "tables", " ".join(sorted(t)))
    head = t["head"]
    print("unitsPerEm", u16(head, 18), "bbox", struct.unpack_from(">hhhh", head, 36), "indexToLocFormat", i16(head, 50))
    print("numGlyphs", u16(t["maxp"], 4), "numberOfHMetrics", u16(t["hhea"], 34))
    ntab, bad, total = checksums(d)
    print("checksums: %d tables, wrong %s, whole file %08x" % (ntab, bad, total))
    mp = t["maxp"]
    print("maxp version %08x" % u32(mp, 0), "maxPoints", u16(mp, 6), "maxContours", u16(mp, 8), "maxCompositePoints",
          u16(mp, 10), "maxCompositeContours", u16(mp, 12), "maxComponentElements", u16(mp, 28), "maxComponentDepth",
          u16(mp, 30))
    print("variable (fvar)", "fvar" in t)
    print("hhea ascender/descender/lineGap", struct.unpack_from(">hhh", t["hhea"], 4))
    if "OS/2" in t:
        o = t["OS/2"]
        ver = u16(o, 0)
        print("OS/2 version", ver, "weightClass", u16(o, 4), "fsSelection", hex(u16(o, 62)),
              "typo asc/desc/gap", struct.unpack_from(">hhh", o, 68), "win asc/desc", struct.unpack_from(">HH", o, 74),
              "xHeight/capHeight", struct.unpack_from(">hh", o, 86) if ver >= 2 else None)
    if "post" in t:
        p = t["post"]
        print("post underline position/thickness", struct.unpack_from(">hh", p, 8), "isFixedPitch", u32(p, 12),
              "italicAngle 16.16", struct.unpack_from(">i", p, 4)[0])
    if "name" in t:
        print("name family/subfamily/full:", [name_string(t, i) for i in (1, 2, 4)],
              "typographic:", [name_string(t, i) for i in (16, 17)])
    print("cmap subtables (platform, encoding, format)", cmap_formats(t["cmap"]))
    n = u16(t["maxp"], 4)
    mapping = cmap_all(t["cmap"], n)
    print("cmap mappings", len(mapping), "sum of code*31+glyph mod 2^32",
          sum(cp * 31 + g for cp, g in mapping.items()) & 0xFFFFFFFF, "first U+%04X" % min(mapping),
          "last U+%04X" % max(mapping))
    print("hmtx sum of advances", sum(advance(t, g) for g in range(n)), "sum of lsb", sum(lsb(t, g) for g in range(n)))

    def outline_facts(gid):
        pts, ends, comps = assemble(t, gid)
        lines, quads = segments(pts, ends)
        b = drawn_bounds(pts, ends)
        return "points %d contours %d components %d lines %d quads %d bounds %s" % (
            len(pts), len(ends), comps, lines, quads, None if b is None else tuple(round(v, 3) for v in b))

    for ch in chars:
        cp = ord(ch) if isinstance(ch, str) else ch
        gid = cmap_lookup(t["cmap"], cp)
        hdr = glyph_header(t, gid)
        print("U+%04X gid %d advance %d lsb %d glyf header %s points %s | %s" % (
            cp, gid, advance(t, gid), lsb(t, gid), hdr, simple_point_count(t, gid) if hdr else None, outline_facts(gid)))
    composites = [g for g in range(n) if (glyph_header(t, g) or (0,))[0] < 0]
    print("composite glyph count", len(composites), "first", composites[:5])
    stats = {}
    tot = [0, 0, 0, 0, 0]
    bsum = [0.0, 0.0, 0.0, 0.0]
    empty = 0
    for gid in range(n):
        pts, ends, comps = assemble(t, gid, stats)
        lines, quads = segments(pts, ends)
        for i, v in enumerate((len(pts), len(ends), comps, lines, quads)):
            tot[i] += v
        b = drawn_bounds(pts, ends)
        if b is None:
            empty += 1
        else:
            for i in range(4):
                bsum[i] += b[i]
    print("all glyphs: points %d contours %d components %d lines %d quads %d, glyphs drawing nothing %d" % (*tot, empty))
    print("all glyphs: sums of drawn bounds xmin/ymin/xmax/ymax", tuple(round(v, 2) for v in bsum))
    print("component flags (uses at the top level of a glyph, first glyphs):",
          {k: (len(v), sorted(set(v))[:3]) for k, v in sorted(stats.items())})
    for name, gids in sorted(stats.items()):
        gid = sorted(set(gids))[0]
        print("  first %s glyph %d: advance %d lsb %d header %s | %s" % (
            name, gid, advance(t, gid), lsb(t, gid), glyph_header(t, gid), outline_facts(gid)))
    differing = [g for g in composites if metrics_source(t, g) != g and (
        advance(t, metrics_source(t, g)), lsb(t, metrics_source(t, g))) != (advance(t, g), lsb(t, g))]
    redirected = [g for g in composites if metrics_source(t, g) != g]
    print("USE_MY_METRICS: glyphs redirected %d (first %s -> %s), with metrics that differ from their own hmtx entry %d %s" % (
        len(redirected), redirected[:3], [metrics_source(t, g) for g in redirected[:3]], len(differing), differing[:5]))
    if "GPOS" in t:
        g = t["GPOS"]
        lookups = gpos_kern_lookups(g)
        ll = u16(g, 8)
        desc = []
        for li in lookups:
            lk = ll + u16(g, ll + 2 + 2 * li)
            kinds = []
            for s in range(u16(g, lk + 4)):
                st = lk + u16(g, lk + 6 + 2 * s)
                styp = u16(g, lk)
                if styp == 9:
                    styp = u16(g, st + 2)
                    st = st + u32(g, st + 4)
                    kinds.append("ext>%d.%d" % (styp, u16(g, st)))
                else:
                    kinds.append("%d.%d" % (styp, u16(g, st)))
            desc.append((li, kinds))
        print("GPOS kern lookups (index, [type.format])", desc)
    for a, b in pairs:
        ga, gb = cmap_lookup(t["cmap"], ord(a)), cmap_lookup(t["cmap"], ord(b))
        print("kerning %s%s (gid %d, %d): GPOS %s, kern table %s" % (a, b, ga, gb, gpos_kern(t, ga, gb), kern_table(t, ga, gb)))
    ascii_glyphs = [cmap_lookup(t["cmap"], c) for c in range(0x21, 0x7F)]
    ascii_glyphs = [g for g in ascii_glyphs if g]
    count = total = weighted = 0
    for i, ga in enumerate(ascii_glyphs):
        for j, gb in enumerate(ascii_glyphs):
            v = gpos_kern(t, ga, gb)
            if v is None:
                v = kern_table(t, ga, gb) or 0
            if v:
                count += 1
                total += v
                weighted += v * (i * 131 + j)
    print("kerning among the %d glyphs of U+0021..U+007E: non-zero pairs %d, sum %d, sum of value*(131*i+j) %d" % (
        len(ascii_glyphs), count, total, weighted))


def path_segments(pts, ends):
    """The drawn contours as (start, segments), a segment being ("L", to) or ("Q", control, to).

    Implied on-curve points are made explicit; the straight line that closes a contour is left out.
    """
    out = []
    first = 0
    for end in ends:
        c = pts[first:end]
        first = end
        if len(c) < 2:
            continue
        if c[0][2]:
            start, rest = (c[0][0], c[0][1]), c[1:]
        elif c[-1][2]:
            start, rest = (c[-1][0], c[-1][1]), c[:-1]
        else:
            start, rest = ((c[0][0] + c[-1][0]) / 2, (c[0][1] + c[-1][1]) / 2), c
        segs = []
        ctrl = None
        for x, y, on in rest:
            if on:
                segs.append(("Q", ctrl, (x, y)) if ctrl is not None else ("L", (x, y)))
                ctrl = None
            else:
                if ctrl is not None:
                    segs.append(("Q", ctrl, ((ctrl[0] + x) / 2, (ctrl[1] + y) / 2)))
                ctrl = (x, y)
        if ctrl is not None:
            segs.append(("Q", ctrl, start))
        out.append((start, segs))
    return out


def exact_bounds(paths):
    """Min/max of the outline itself: segment end points plus the interior extrema of the curves."""
    xs, ys = [], []
    for start, segs in paths:
        cur = start
        xs.append(cur[0])
        ys.append(cur[1])
        for seg in segs:
            to = seg[-1]
            if seg[0] == "Q":
                ctrl = seg[1]
                for axis, acc in ((0, xs), (1, ys)):
                    a, c, b = cur[axis], ctrl[axis], to[axis]
                    d = a - 2 * c + b
                    if d != 0:
                        t = (a - c) / d
                        if 0 < t < 1:
                            acc.append((1 - t) * (1 - t) * a + 2 * (1 - t) * t * c + t * t * b)
            xs.append(to[0])
            ys.append(to[1])
            cur = to
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def signed_area(paths):
    """The exact signed area of the outline (positive counter-clockwise), curves included, by Green's theorem
    integrated numerically-free: each quadratic contributes its chord plus two thirds of its control triangle."""
    def cross(p, q):
        return p[0] * q[1] - p[1] * q[0]

    twice = 0.0
    for start, segs in paths:
        cur = start
        for seg in segs:
            to = seg[-1]
            if seg[0] == "Q":
                c = seg[1]
                twice += (2 * cross(cur, c) + 2 * cross(c, to) + cross(cur, to)) / 3
            else:
                twice += cross(cur, to)
            cur = to
        twice += cross(cur, start)
    return twice / 2


def gpos_pair_lookups(g):
    """For each lookup of the kern feature that has pair subtables: the offsets of those subtables."""
    out = []
    ll = u16(g, 8)
    for li in gpos_kern_lookups(g):
        lk = ll + u16(g, ll + 2 + 2 * li)
        ltype = u16(g, lk)
        subs = []
        for k in range(u16(g, lk + 4)):
            st = lk + u16(g, lk + 6 + 2 * k)
            styp = ltype
            if styp == 9:
                styp = u16(g, st + 2)
                st = st + u32(g, st + 4)
            if styp == 2:
                subs.append(st)
        if subs:
            out.append(subs)
    return out


def kerning_pairs(t, glyphs):
    """(source, [(left, right, value)]) of every non-zero kerning among glyphs, in the order of glyphs."""
    out = []
    lookups = gpos_pair_lookups(t["GPOS"]) if "GPOS" in t else []
    if lookups:
        g = t["GPOS"]
        for left in glyphs:
            covering = [[st for st in subs if coverage_index(g, st + u16(g, st + 2), left) is not None] for subs in lookups]
            if not any(covering):
                continue
            for right in glyphs:
                total = 0
                for subs in covering:
                    for st in subs:
                        ok, v = pair_subtable(g, st, left, right)
                        if ok:
                            total += v
                            break
                if total:
                    out.append((left, right, total))
        return "gpos", out
    if "kern" in t and u16(t["kern"], 0) == 0:
        for left in glyphs:
            for right in glyphs:
                v = kern_table(t, left, right)
                if v:
                    out.append((left, right, v))
        return "kern", out
    return "none", out


def fmt_num(v):
    """At most four decimals, without trailing zeros."""
    text = ("%.4f" % v).rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def fmt_box(b):
    return "-" if b is None else " ".join(fmt_num(v) for v in b)


def write_fixture(path, outdir):
    """Writes <outdir>/<font name>.txt, the facts crates/io/src/ttf/tests.rs compares against.

    Lines (space separated; "-" for a value the font does not have):

        font <file name> <size in bytes>
        head <unitsPerEm> <xMin> <yMin> <xMax> <yMax> <indexToLocFormat>
        maxp <numGlyphs> <maxPoints> <maxContours> <maxCompositePoints> <maxCompositeContours>
             <maxComponentElements> <maxComponentDepth>
        hhea <ascender> <descender> <lineGap> <numberOfHMetrics>
        os2 <version> <usWeightClass> <sTypoAscender> <sTypoDescender> <sTypoLineGap> <sxHeight> <sCapHeight>
        post <underlinePosition> <underlineThickness> <isFixedPitch> <italicAngle as 16.16>
        variable <0|1>
        checksums <number of tables> <number of wrong table checksums> <whole-file sum, hex>
        name <id> <string>                          (ids 1, 2 and 4, when present)
        cmap <number of mappings>
        c <code point, hex> <glyph>                 (every mapping, by code point)
        glyphs <number of glyphs>
        g <glyph> <advance> <lsb> <metrics source glyph> <kind: s simple, c composite, e empty>
          <stored xMin yMin xMax yMax | -> <points> <contours> <components> <lines> <quads>
          <control bounds: 4 numbers | -> <exact bounds: 4 numbers | - | = when equal to the control
          bounds> <signed area>                                                            (one line)
        kern <gpos|kern|none> <number of glyphs considered> <number of non-zero pairs>
        k <left glyph> <right glyph> <value>        (by position of the glyphs in increasing id order)
    """
    d = open(path, "rb").read()
    t = tables(d)
    name = path.replace("\\", "/").split("/")[-1]
    lines = ["# Written by tools/ttf_crosscheck.py --fixtures; do not edit by hand.",
             "font %s %d" % (name, len(d))]
    head, mp, hhea = t["head"], t["maxp"], t["hhea"]
    lines.append("head %d %d %d %d %d %d" % ((u16(head, 18),) + struct.unpack_from(">hhhh", head, 36) + (i16(head, 50),)))
    n = u16(mp, 4)
    lines.append("maxp %d %d %d %d %d %d %d" % (n, u16(mp, 6), u16(mp, 8), u16(mp, 10), u16(mp, 12), u16(mp, 28), u16(mp, 30)))
    lines.append("hhea %d %d %d %d" % (struct.unpack_from(">hhh", hhea, 4) + (u16(hhea, 34),)))
    if "OS/2" in t:
        o = t["OS/2"]
        ver = u16(o, 0)
        heights = "%d %d" % struct.unpack_from(">hh", o, 86) if ver >= 2 else "- -"
        lines.append("os2 %d %d %d %d %d %s" % ((ver, u16(o, 4)) + struct.unpack_from(">hhh", o, 68) + (heights,)))
    if "post" in t:
        po = t["post"]
        lines.append("post %d %d %d %d" % (struct.unpack_from(">hh", po, 8) + (u32(po, 12), struct.unpack_from(">i", po, 4)[0])))
    lines.append("variable %d" % ("fvar" in t))
    ntab, bad, total = checksums(d)
    lines.append("checksums %d %d %08x" % (ntab, len(bad), total))
    if "name" in t:
        for nid in (1, 2, 4):
            text = name_string(t, nid)
            if text is not None:
                lines.append("name %d %s" % (nid, text))
    mapping = cmap_all(t["cmap"], n)
    lines.append("cmap %d" % len(mapping))
    for cp in sorted(mapping):
        lines.append("c %X %d" % (cp, mapping[cp]))
    lines.append("glyphs %d" % n)
    for gid in range(n):
        hdr = glyph_header(t, gid)
        pts, ends, comps = assemble(t, gid)
        nlines, nquads = segments(pts, ends)
        paths = path_segments(pts, ends)
        # Two independent counts of the same thing must agree.
        assert nlines == sum(1 for _s, sg in paths for x in sg if x[0] == "L"), gid
        assert nquads == sum(1 for _s, sg in paths for x in sg if x[0] == "Q"), gid
        kind = "e" if hdr is None else "c" if hdr[0] < 0 else "s"
        control, exact = drawn_bounds(pts, ends), exact_bounds(paths)
        lines.append("g %d %d %d %d %s %s %d %d %d %d %d %s %s %s" % (
            gid, advance(t, gid), lsb(t, gid), metrics_source(t, gid), kind,
            "-" if hdr is None else "%d %d %d %d" % hdr[1:], len(pts), len(ends), comps, nlines, nquads,
            fmt_box(control), "=" if exact is not None and exact == control else fmt_box(exact),
            fmt_num(signed_area(paths))))
    codes = list(range(0x20, 0x7F)) + list(range(0xA0, 0x100))
    glyphs = sorted({mapping[cp] for cp in codes if cp in mapping})
    source, pairs = kerning_pairs(t, glyphs)
    lines.append("kern %s %d %d" % (source, len(glyphs), len(pairs)))
    for left, right, value in pairs:
        lines.append("k %d %d %d" % (left, right, value))
    out = outdir.rstrip("/\\") + "/" + name.rsplit(".", 1)[0] + ".txt"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out, len(lines), "lines:", len(mapping), "mappings,", n, "glyphs,", len(pairs), "kerning pairs from", source)


def mac_roman_check():
    """A checksum of the upper half of Mac Roman as Python's codec defines it."""
    chars = bytes(range(0x80, 0x100)).decode("mac_roman")
    return sum((0x80 + i) * ord(c) for i, c in enumerate(chars))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fixtures":
        for font in sys.argv[3:]:
            write_fixture(font, sys.argv[2])
        sys.exit(0)
    print("mac_roman 0x80..0xFF: sum of byte*codepoint", mac_roman_check())
    PAIRS = [("A", "V"), ("T", "o"), ("V", "A"), ("A", "A"), ("L", "T"), ("W", "a"), ("r", "."), ("H", "H"), ("f", "f")]
    for font in sys.argv[1:]:
        extra = [0xE000, 0xE001, 0xE038, 0xE100, 0xE2FF] if "Lucide" in font else []
        report(font, PAIRS, ["A", " ", "V", "T", "o", "g", "M", "i", "é", "Å"] + extra)
