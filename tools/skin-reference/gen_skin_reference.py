#!/usr/bin/env python3
"""Builds data/test-scenes/skin-reference.json from the International Skin Spectra Archive (ISSA).

Source: Lu Y., Xiao K., Pointer M., He R., Zhou S., Nasseraldin A., Sueeprasan S., Gao C., Li C.,
Sohaib A., He Y., Mizokami Y., Wang M., Perales Romero E., Zardawi F., Gill L., Liao N., Wuerger S.,
"The International Skin Spectra Archive (ISSA): a multicultural human skin phenotype and colour
spectra collection", Scientific Data 12, 487 (2025), doi:10.1038/s41597-025-04857-5. Data: figshare
doi:10.6084/m9.figshare.28228571.v4, file ISSA_17_Jan_2025_Yan_Lu.xlsx (9 320 520 bytes, MD5
67e15398f96dd9af08b5d209d2522e19 as listed by figshare). Licence CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Standard library only. The workbook is untrusted input: the file and every zip member are size- and
ratio-limited, XML with a DOCTYPE is refused, the row and column counts are bounded, the coding sheet
must name the columns this script expects, and every record must carry finite values inside
documented ranges (reflectance 0..150 %, L* 0..100, |a*|, |b*| <= 128) and a complete spectrum for
its stated wavelength range. Any violation aborts with a message; nothing is silently dropped.

Usage (run from anywhere; the workbook is passed as a path, never imported or executed):

    python -I tools/skin-reference/gen_skin_reference.py \
        <ISSA_17_Jan_2025_Yan_Lu.xlsx> <retrieved YYYY-MM-DD> data/test-scenes/skin-reference.json

The output is deterministic for a given workbook (fixed rounding, fixed key order, no timestamps
other than the retrieval date passed in). Its members are described in FORMAT_NOTE.
"""
import hashlib
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

MAX_FILE_BYTES = 64 << 20
MAX_MEMBERS = 200
MAX_MEMBER_BYTES = 512 << 20
MAX_RATIO = 200
MAX_SHARED_STRINGS = 1_000_000
MAX_ROWS = 200_000
MAX_COLUMN = 1000
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Columns A..L of the data sheet as the coding sheet ("COLUMNS") names them.
COLUMNS = ["Record number", "Origin of data", "Subject number", "Ethnicity/Group", "Gender", "Age group",
           "Body location", "Instrument type", "SPIN", "Start wavelength", "End wavelength",
           "Wavelength interval"]
# Code tables of the coding sheet: header -> (code column, name column).
CODE_TABLES = {"ORIGIN": 3, "ETHNICITY": 6, "GENDER": 9, "AGE GROUP": 12, "BODY LOCATION": 15}
WAVELENGTHS = list(range(400, 701, 10))
# ITA categories (Chardon, Dechamps, Hourseau 1991; Del Bino et al. 2006): (id, lower bound, upper bound).
ITA_BINS = [("very-light", 55.0, 180.0), ("light", 41.0, 55.0), ("intermediate", 28.0, 41.0),
            ("tan", 10.0, 28.0), ("brown", -30.0, 10.0), ("dark", -180.0, -30.0)]
LIGHTNESS_EDGES = list(range(20, 81, 5))
MIN_BIN_RECORDS = 30

FORMAT_NOTE = (
    "Aggregated from ISSA by tools/skin-reference/gen_skin_reference.py. lab_* are CIELAB (CIE 1931 2deg "
    "observer, illuminant D65) as computed by the dataset authors from each spectrum (specular component "
    "included). lab_mean is the mean, lab_sd the sample SD, lab_pNN the NNth percentile of each coordinate "
    "separately (linear interpolation). subjects counts distinct (origin, subject number) pairs. ita_deg = atan2(L* - 50, b*) in degrees. reflectance_mean is the mean "
    "reflectance factor (0..1) at wavelengths_nm. In ita_bins, 'subsets' holds the mean Lab of the records "
    "at or below the bin's 10th percentile (low) or at or above its 90th percentile (high) of L*, a* or b*: "
    "measured swatches at the edges of each bin. Bins with fewer than 30 records are omitted.")


def fail(message):
    sys.exit("gen_skin_reference: " + message)


def read_member(z, name):
    try:
        info = z.getinfo(name)
    except KeyError:
        fail("missing zip member " + name)
    if info.file_size > MAX_MEMBER_BYTES:
        fail(name + " is too large")
    if info.compress_size > 0 and info.file_size / info.compress_size > MAX_RATIO:
        fail(name + " has an implausible compression ratio")
    data = z.read(name)
    if len(data) != info.file_size:
        fail(name + " size mismatch")
    if b"<!DOCTYPE" in data[:4096] or b"<!ENTITY" in data:
        fail(name + " contains a DOCTYPE or entity declaration")
    return data


def col_index(ref):
    m = re.match(r"([A-Z]{1,3})[0-9]+$", ref or "")
    if not m:
        fail("bad cell reference %r" % ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    if n > MAX_COLUMN:
        fail("column beyond the limit: " + ref)
    return n - 1


def shared_strings(z):
    root = ET.fromstring(read_member(z, "xl/sharedStrings.xml"))
    items = root.findall(M + "si")
    if len(items) > MAX_SHARED_STRINGS:
        fail("too many shared strings")
    return ["".join(t.text or "" for t in si.iter(M + "t")) for si in items]


def sheet_rows(z, name, shared):
    root = ET.fromstring(read_member(z, name))
    rows = root.iter(M + "row")
    out = []
    for row in rows:
        if len(out) >= MAX_ROWS:
            fail(name + " has too many rows")
        cells = {}
        for c in row.findall(M + "c"):
            v = c.find(M + "v")
            if v is None:
                inline = c.find(M + "is")
                if inline is None:
                    continue
                value = "".join(t.text or "" for t in inline.iter(M + "t"))
            else:
                value = v.text or ""
                if c.get("t") == "s":
                    if not value.isdigit() or int(value) >= len(shared):
                        fail("shared string index out of range in " + name)
                    value = shared[int(value)]
            cells[col_index(c.get("r"))] = value.strip()
        out.append(cells)
    return out


def number(text, what, lo, hi):
    try:
        v = float(text)
    except (TypeError, ValueError):
        fail("%s is not a number: %r" % (what, text))
    if not math.isfinite(v) or v < lo or v > hi:
        fail("%s out of range %s..%s: %r" % (what, lo, hi, v))
    return v


def coding(rows):
    """The column names and the code tables of the coding sheet."""
    header = next((i for i, r in enumerate(rows) if r.get(0) == "COLUMNS"), None)
    if header is None:
        fail("coding sheet has no COLUMNS header")
    for key, col in CODE_TABLES.items():
        if rows[header].get(col) != key:
            fail("coding sheet column %d is not %s" % (col, key))
    names = []
    tables = {key: {} for key in CODE_TABLES}
    for r in rows[header + 1:]:
        if r.get(1):
            names.append(r[1])
        for key, col in CODE_TABLES.items():
            if r.get(col) and r.get(col + 1):
                tables[key][r[col]] = r[col + 1]
    if names[:len(COLUMNS)] != COLUMNS:
        fail("coding sheet columns differ from the expected A..L: %r" % names[:len(COLUMNS)])
    return tables


def data_layout(rows):
    """Column indices of the wavelengths and of L*, a*, b* from the data sheet's header row."""
    for r in rows:
        values = {v: k for k, v in r.items()}
        if "L*" in values and "a*" in values and "b*" in values and "360" in values:
            wl = {}
            for k, v in r.items():
                if v.isdigit() and 300 <= int(v) <= 800:
                    wl[int(v)] = k
            if any(w not in wl for w in range(360, 741, 10)):
                fail("data header lacks a wavelength column")
            return wl, values["L*"], values["a*"], values["b*"]
    fail("data sheet has no header row with L*, a*, b*")


def records(rows, tables, layout):
    wl, cl, ca, cb = layout
    out = []
    for r in rows:
        rec_id = r.get(0, "")
        if not rec_id.isdigit() or r.get(3) is None:
            continue
        what = "record " + rec_id
        group, site, gender, age = r.get(3), r.get(6), r.get(4), r.get(5)
        if group not in tables["ETHNICITY"]:
            fail(what + ": unknown group %r" % group)
        if site not in tables["BODY LOCATION"]:
            fail(what + ": unknown body location %r" % site)
        if r.get(1) not in tables["ORIGIN"]:
            fail(what + ": unknown origin %r" % r.get(1))
        if gender not in (None, "") and gender not in tables["GENDER"]:
            fail(what + ": unknown gender %r" % gender)
        if age not in (None, "") and age not in tables["AGE GROUP"]:
            fail(what + ": unknown age group %r" % age)
        if r.get(7) != "SP" or r.get(8) != "SCI":
            fail(what + ": not a specular-included spectrophotometer record")
        start = int(number(r.get(9), what + " start", 360, 400))
        end = int(number(r.get(10), what + " end", 700, 740))
        if number(r.get(11), what + " step", 10, 10) != 10 or start % 10 or end % 10:
            fail(what + ": unexpected wavelength grid")
        spectrum = {}
        for w in range(start, end + 1, 10):
            spectrum[w] = number(r.get(wl[w]), "%s %d nm" % (what, w), 0.0, 150.0) / 100.0
        if any(w not in spectrum for w in WAVELENGTHS):
            fail(what + ": spectrum does not cover 400..700 nm")
        lab = [number(r.get(cl), what + " L*", 0, 100), number(r.get(ca), what + " a*", -128, 128),
               number(r.get(cb), what + " b*", -128, 128)]
        out.append({"origin": r[1], "subject": r.get(2, ""), "group": group, "site": site,
                    "gender": gender or "", "age": age or "", "lab": lab,
                    "ita": math.degrees(math.atan2(lab[0] - 50.0, lab[2])),
                    "refl": [spectrum[w] for w in WAVELENGTHS]})
    return out


def percentile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def r3(v):
    return [round(x, 3) for x in v]


def lab_stats(sel):
    out = {"n": len(sel), "subjects": len({(s["origin"], s["subject"]) for s in sel})}
    cols = [[s["lab"][i] for s in sel] for i in range(3)]
    mean = [sum(c) / len(c) for c in cols]
    out["lab_mean"] = r3(mean)
    out["lab_sd"] = r3([math.sqrt(sum((x - m) ** 2 for x in c) / (len(c) - 1)) for c, m in zip(cols, mean)])
    sorted_cols = [sorted(c) for c in cols]
    for q in (5, 25, 50, 75, 95):
        out["lab_p%02d" % q] = r3([percentile(c, q) for c in sorted_cols])
    out["ita_deg_mean"] = round(sum(s["ita"] for s in sel) / len(sel), 2)
    return out


def counts(sel, key):
    out = {}
    for s in sel:
        out[s[key]] = out.get(s[key], 0) + 1
    return {k: out[k] for k in sorted(out)}


def mean_reflectance(sel):
    return [round(sum(s["refl"][i] for s in sel) / len(sel), 4) for i in range(len(WAVELENGTHS))]


def subsets(sel):
    out = {}
    for i, axis in enumerate(("l", "a", "b")):
        values = sorted(s["lab"][i] for s in sel)
        lo, hi = percentile(values, 10), percentile(values, 90)
        for side, pick in (("low", [s for s in sel if s["lab"][i] <= lo]), ("high", [s for s in sel if s["lab"][i] >= hi])):
            out["%s_%s" % (axis, side)] = {"n": len(pick), "lab_mean": r3([sum(s["lab"][k] for s in pick) / len(pick) for k in range(3)])}
    return out


def dump(value, indent=0):
    """JSON with objects indented and arrays of scalars on one line (deterministic)."""
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ['%s "%s": %s' % (pad, k, dump(v, indent + 1)) for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(dump(v) for v in value) + "]"
        return "[\n" + ",\n".join(pad + " " + dump(v, indent + 1) for v in value) + "\n" + pad + "]"
    if isinstance(value, bool) or value is None:
        return {True: "true", False: "false", None: "null"}[value]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            fail("non-finite number in the output")
        return repr(value)
    text = str(value)
    if any(ord(ch) < 0x20 or ord(ch) > 0x7e for ch in text):
        fail("non-ASCII or control character in output text %r" % text)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    if len(sys.argv) != 4 or not re.match(r"^\d{4}-\d{2}-\d{2}$", sys.argv[2]):
        fail("usage: gen_skin_reference.py <xlsx> <retrieved YYYY-MM-DD> <out.json>")
    path, retrieved, out_path = sys.argv[1:]
    with open(path, "rb") as f:
        raw = f.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        fail("workbook larger than %d bytes" % MAX_FILE_BYTES)
    sha256 = hashlib.sha256(raw).hexdigest()
    md5 = hashlib.md5(raw).hexdigest()
    z = zipfile.ZipFile(path)
    if len(z.infolist()) > MAX_MEMBERS:
        fail("too many zip members")
    shared = shared_strings(z)
    tables = coding(sheet_rows(z, "xl/worksheets/sheet1.xml", shared))
    data_rows = sheet_rows(z, "xl/worksheets/sheet2.xml", shared)
    recs = records(data_rows, tables, data_layout(data_rows))
    if not recs:
        fail("no records")

    groups = {}
    for code in sorted({s["group"] for s in recs}):
        sel = [s for s in recs if s["group"] == code]
        groups[code] = {"name": tables["ETHNICITY"][code], "records": len(sel),
                        "subjects": len({(s["origin"], s["subject"]) for s in sel})}
    origins = {}
    for code in sorted({s["origin"] for s in recs}, key=int):
        origins[code] = {"name": tables["ORIGIN"][code], "records": sum(1 for s in recs if s["origin"] == code)}
    site_name = lambda code: tables["BODY LOCATION"][code].lower().replace(" ", "-")

    ita_bins = []
    for name, lo, hi in ITA_BINS:
        sel = [s for s in recs if lo < s["ita"] <= hi]
        if len(sel) < MIN_BIN_RECORDS:
            continue
        b = {"id": name, "ita_min_deg": lo, "ita_max_deg": hi}
        b.update(lab_stats(sel))
        b["groups"] = counts(sel, "group")
        b["subsets"] = subsets(sel)
        b["reflectance_mean"] = mean_reflectance(sel)
        ita_bins.append(b)

    lightness_bins = []
    for lo, hi in zip(LIGHTNESS_EDGES, LIGHTNESS_EDGES[1:]):
        sel = [s for s in recs if lo <= s["lab"][0] < hi]
        if len(sel) < MIN_BIN_RECORDS:
            continue
        b = {"l_min": lo, "l_max": hi}
        b.update(lab_stats(sel))
        b["groups"] = counts(sel, "group")
        b["reflectance_mean"] = mean_reflectance(sel)
        lightness_bins.append(b)

    sites = []
    for code in sorted({s["site"] for s in recs}, key=int):
        sel = [s for s in recs if s["site"] == code]
        b = {"id": site_name(code)}
        b.update(lab_stats(sel))
        sites.append(b)

    covered = ", ".join("%s (%s)" % (g["name"], code) for code, g in groups.items())
    doc = {
        "format": "humanart.skin-reference",
        "version": 1,
        "note": FORMAT_NOTE,
        "source": {
            "title": "The International Skin Spectra Archive (ISSA): a multicultural human skin phenotype and colour spectra collection",
            "authors": "Lu Y., Xiao K., Pointer M., He R., Zhou S., Nasseraldin A., Sueeprasan S., Gao C., Li C., Sohaib A., He Y., Mizokami Y., Wang M., Perales Romero E., Zardawi F., Gill L., Liao N., Wuerger S.",
            "journal": "Scientific Data 12, 487 (2025)",
            "article_doi": "10.1038/s41597-025-04857-5",
            "data_doi": "10.6084/m9.figshare.28228571.v4",
            "file": path.replace("\\", "/").rsplit("/", 1)[-1],
            "file_bytes": len(raw),
            "file_sha256": sha256,
            "file_md5": md5,
            "retrieved": retrieved,
        },
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Contains statistics derived from the International Skin Spectra Archive (ISSA), Lu Y. et al., Sci Data 12, 487 (2025), doi:10.1038/s41597-025-04857-5, data doi:10.6084/m9.figshare.28228571.v4, licensed under CC BY 4.0.",
        "changes": "Aggregated into bin statistics and mean spectra (400-700 nm); no individual record is reproduced.",
        "measurement": {
            "instrument": "spectrophotometer (all records)",
            "geometry": "specular component included (SCI)",
            "spectra": "10 nm steps; %d records 400-700 nm, %d records 360-740 nm; this file uses 400-700 nm" % (
                sum(1 for s in data_rows if s.get(9) == "400" and str(s.get(0, "")).isdigit()),
                sum(1 for s in data_rows if s.get(9) == "360" and str(s.get(0, "")).isdigit())),
            "colorimetry": "CIELAB, CIE 1931 2deg observer, illuminant D65, computed by the dataset authors",
        },
        "population": {
            "records": len(recs),
            "subjects": len({(s["origin"], s["subject"]) for s in recs}),
            "groups": groups,
            "origins": origins,
            "gender": {tables["GENDER"].get(k, "unknown"): v for k, v in counts(recs, "gender").items()},
            "age_groups": {tables["AGE GROUP"].get(k, "unknown"): v for k, v in counts(recs, "age").items()},
            "body_sites": {site_name(k): v for k, v in sorted(counts(recs, "site").items(), key=lambda kv: int(kv[0]))},
            "coverage_note": "Covers the self-reported groups " + covered + "; groups not listed are not "
                             "represented, and no group is broken down by region. No Fitzpatrick phototype is "
                             "recorded: the bins use ITA categories instead.",
        },
        "wavelengths_nm": WAVELENGTHS,
        "ita_bins": ita_bins,
        "lightness_bins": lightness_bins,
        "body_sites": sites,
    }
    text = dump(doc) + "\n"
    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print("records", len(recs), "ita bins", len(ita_bins), "lightness bins", len(lightness_bins), "bytes", len(text))


if __name__ == "__main__":
    main()
