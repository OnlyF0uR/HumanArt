#!/usr/bin/env python3
"""Builds data/anthropometry/ansur2.json from the public ANSUR II working databases.

Source: 2012 US Army Anthropometric Survey (ANSUR II), working databases released by the US Army
Natick Soldier Research, Development and Engineering Center, distributed by the OPEN Design Lab,
Penn State (https://www.openlab.psu.edu/ansur2/). Methods and definitions: Gordon C.C. et al.,
"2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary Statistics",
NATICK/TR-15/007, December 2014.

Standard library only. The CSVs are untrusted input: every file is size-limited, every row must have
exactly the header's column count, every measurement must be a decimal integer inside a documented
range, and the Gender column must match the file. Any violation aborts with a message; nothing is
silently dropped.

Usage (run from anywhere; the CSVs are passed as paths, never imported):

    python -I tools/anthropometry/ansur2_summary.py \
        <ANSUR_II_MALE_Public.csv> <ANSUR_II_FEMALE_Public.csv> <retrieved YYYY-MM-DD> \
        data/anthropometry/ansur2.json

The output format is described in the module constant FORMAT_NOTE and in the file's own "format"
member.
"""
import csv
import hashlib
import io
import json
import math
import sys

MAX_CSV_BYTES = 16 << 20  # each public CSV is about 2 MB
MAX_ROWS = 20000  # the survey has 4082 + 1986 subjects
# Every length is in millimetres; weightkg is stored in tenths of a kilogram (MFR, 14 Apr 2017).
VALUE_RANGE = (1, 30000)
# Columns stored in tenths of their unit. weightkg is documented as kg*10 by the memorandum.
# interpupillarybreadth is NOT documented: the memorandum says every length is in mm, but the CSV
# values (median 640 male, 620 female) are tenths of a millimetre, as the published percentiles of
# TR-15/007 (median 6.4 / 6.2 cm) show. Checked when this file was written: P5/P50/P95 of 58 columns
# match the report's tables within 0.5 mm, and the P50 of the other 35 within 1 %.
TENTHS = {"weightkg", "interpupillarybreadth"}
AGE_RANGE = (15, 80)

PAGE_URL = "https://www.openlab.psu.edu/ansur2/"
MALE_URL = "https://tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv"
FEMALE_URL = "https://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv"
MFR_URL = "https://tools.openlab.psu.edu/publicData/ANSURII-MFR.pdf"
REPORT_URL = "https://tools.openlab.psu.edu/publicData/ANSURII-TR15-007.pdf"

FORMAT_NOTE = (
    "UTF-8 JSON, one object. 'source' names the data and its licence. 'variables' lists the "
    "measurements given a definition (id = CSV column, title and first sentence of the definition "
    "from NATICK/TR-15/007, unit). 'stats.male' and 'stats.female' map every measurement column "
    "(all 93) plus the derived ids 'bmi', 'waist_to_height', 'waist_to_hip' and 'age' to "
    "{n, mean, sd, min, p5, p50, p95, max}; lengths in mm, weightkg in kg, bmi in kg/m^2, ratios "
    "unitless, age in years. sd is the sample standard deviation (n-1). Percentiles use linear "
    "interpolation between order statistics (Hyndman-Fan type 7). 'sex_midpoint_mean' is "
    "(male mean + female mean) / 2 for every id: NOT a population statistic (see "
    "source.population_note). 'correlation' holds Pearson correlation matrices per sex over "
    "correlation.variables, row-major, 3 decimals. 'population' describes who was measured."
)

# Measurements given a definition in the output: (CSV column, report title, first sentence of the
# definition in NATICK/TR-15/007, extracted from the report PDF; inter-word spacing artefacts of the
# PDF text layer corrected, wording unchanged).
DEFINITIONS = [
    ("stature", "STATURE",
     "The vertical distance from a standing surface to the top of the head is measured with an anthropometer."),
    ("weightkg", "WEIGHT",
     "The weight of the participant is taken to the nearest tenth of a kilogram."),
    ("acromialheight", "ACROMIAL HEIGHT",
     "The vertical distance between a standing surface and the right acromion landmark is measured with an anthropometer."),
    ("cervicaleheight", "CERVICALE HEIGHT",
     "The vertical distance between a standing surface and the cervicale landmark is measured with an anthropometer."),
    ("suprasternaleheight", "SUPRASTERNALE HEIGHT",
     "The vertical distance between a standing surface and the suprasternale landmark is measured with an anthropometer."),
    ("chestheight", "CHEST HEIGHT",
     "The vertical distance between a standing surface and the chest point anterior landmark is measured with an anthropometer."),
    ("tenthribheight", "TENTH RIB HEIGHT",
     "The vertical distance between a standing surface and the tenth rib landmark is measured with an anthropometer."),
    ("waistheightomphalion", "WAIST HEIGHT (OMPHALION)",
     "The vertical distance between a standing surface and the anterior omphalion landmark is measured with an anthropometer."),
    ("iliocristaleheight", "ILIOCRISTALE HEIGHT",
     "The vertical distance between a standing surface and the right iliocristale landmark is measured with an anthropometer."),
    ("trochanterionheight", "TROCHANTERION HEIGHT",
     "The vertical distance between a standing surface and the trochanterion landmark is measured with an anthropometer."),
    ("crotchheight", "CROTCH HEIGHT",
     "The vertical distance between the standing surface and the crotch is measured with an anthropometer."),
    ("kneeheightmidpatella", "KNEE HEIGHT, MIDPATELLA",
     "The vertical distance between a standing surface and the midpatella landmark is measured with an anthropometer."),
    ("lateralfemoralepicondyleheight", "LATERAL FEMORAL EPICONDYLE HEIGHT",
     "The vertical distance between a standing surface and the standing lateral femoral epicondyle landmark is measured with an anthropometer."),
    ("tibialheight", "TIBIAL HEIGHT",
     "The vertical distance between a standing surface and the tibiale landmark is measured with an anthropometer."),
    ("lateralmalleolusheight", "LATERAL MALLEOLUS HEIGHT",
     "The vertical distance between a standing surface and the lateral malleolus landmark is measured with a modified height gauge."),
    ("sittingheight", "SITTING HEIGHT",
     "The vertical distance between a sitting surface and the top of the head is measured with an anthropometer."),
    ("span", "SPAN",
     "The distance between the tips of the middle fingers (dactylion III) of the horizontally outstretched arms is measured on a wall chart."),
    ("acromionradialelength", "ACROMION-RADIALE LENGTH",
     "The distance between the right acromion landmark and the radiale landmark is measured with a beam caliper held parallel to the long axis of the arm."),
    ("shoulderelbowlength", "SHOULDER-ELBOW LENGTH",
     "The distance between the right acromion landmark and the olecranon bottom landmark is measured with a beam caliper parallel to the long axis of the upper arm."),
    ("radialestylionlength", "RADIALE-STYLION LENGTH",
     "The distance between the radiale landmark and the stylion landmark is measured with a beam caliper held parallel to the long axis of the forearm."),
    ("forearmhandlength", "FOREARM-HAND LENGTH",
     "The horizontal distance between the olecranon rear landmark and the dactylion III landmark is measured with a beam caliper."),
    ("buttockkneelength", "BUTTOCK-KNEE LENGTH",
     "The horizontal distance between a buttock plate placed at the most posterior point on either buttock and the knee point, anterior landmark, is measured with an anthropometer."),
    ("biacromialbreadth", "BIACROMIAL BREADTH",
     "The distance between the right and left acromion landmarks is measured with a beam caliper."),
    ("bideltoidbreadth", "BIDELTOID BREADTH",
     "The maximum horizontal distance between the lateral margins of the upper arms on the deltoid muscles is measured with a beam caliper."),
    ("chestbreadth", "CHEST BREADTH",
     "The maximum horizontal breadth of the chest at the level of the chest point anterior landmark is measured with a beam caliper."),
    ("chestdepth", "CHEST DEPTH",
     "The horizontal distance between the chest point anterior landmark and the back at the same level is measured with a beam caliper."),
    ("waistbreadth", "WAIST BREADTH",
     "The horizontal breadth of the waist at the level of omphalion is measured with a beam caliper."),
    ("waistdepth", "WAIST DEPTH",
     "The horizontal distance between the anterior and posterior omphalion landmarks is measured with a beam caliper."),
    ("bicristalbreadth", "BICRISTAL BREADTH",
     "The straight-line distance between the right and left iliocristale landmarks is measured with a beam caliper."),
    ("hipbreadth", "HIP BREADTH",
     "The horizontal distance between the lateral buttock landmarks is measured with a beam caliper."),
    ("buttockdepth", "BUTTOCK DEPTH",
     "The horizontal depth of the torso at the level of the buttock point, posterior and right lateral landmarks, is measured using a beam caliper."),
    ("chestcircumference", "CHEST CIRCUMFERENCE",
     "The maximum horizontal circumference of the chest at the level of chest point, anterior is measured with a tape."),
    ("waistcircumference", "WAIST CIRCUMFERENCE (OMPHALION)",
     "The horizontal circumference of the waist, passing over all omphalion landmarks, is measured with a tape."),
    ("buttockcircumference", "BUTTOCK CIRCUMFERENCE",
     "The horizontal circumference of the trunk at the level of the buttock point, posterior, right and left lateral landmarks, is measured with a tape."),
    ("thighcircumference", "THIGH CIRCUMFERENCE",
     "The circumference of the right thigh at the gluteal furrow landmark is measured with a tape."),
    ("lowerthighcircumference", "LOWER THIGH CIRCUMFERENCE",
     "The horizontal circumference of the right thigh at the level of the suprapatella landmark is measured with a tape."),
    ("calfcircumference", "CALF CIRCUMFERENCE",
     "The maximum horizontal circumference of the right calf is measured with a tape."),
    ("anklecircumference", "ANKLE CIRCUMFERENCE",
     "The minimum horizontal circumference of the right ankle is measured with a tape."),
    ("bicepscircumferenceflexed", "BICEPS CIRCUMFERENCE, FLEXED",
     "The circumference of the right upper arm around the flexed biceps brachii muscle at the biceps point landmark is measured with a tape held perpendicular to the long axis of the upper arm."),
    ("forearmcircumferenceflexed", "FOREARM CIRCUMFERENCE, FLEXED",
     "The circumference of the flexed right forearm is measured with a tape passing across the crease at the juncture between the upper arm and the forearm."),
    ("wristcircumference", "WRIST CIRCUMFERENCE",
     "The circumference of the wrist, perpendicular to the long axis of the forearm, is measured with a tape passing over the stylion landmark."),
    ("neckcircumference", "NECK CIRCUMFERENCE",
     "The circumference of the neck at the level of the infrathyroid landmark (Adam's apple) is measured with a tape."),
    ("neckcircumferencebase", "NECK CIRCUMFERENCE, BASE",
     "The circumference of the base of the neck is measured with a tape, passing over the drawn lateral and anterior neck landmarks."),
    ("shouldercircumference", "SHOULDER CIRCUMFERENCE",
     "The circumference of the shoulders at the level of the right and left deltoid point landmarks is measured with a tape."),
    ("abdominalextensiondepthsitting", "ABDOMINAL EXTENSION DEPTH, SITTING",
     "The horizontal distance between the abdominal point anterior and the back at the same level is measured with a beam caliper."),
    ("handlength", "HAND LENGTH",
     "The length of the right hand between the stylion landmark on the wrist and the tip of the middle finger is measured with a Poech sliding caliper."),
    ("handbreadth", "HAND BREADTH",
     "The breadth of the right hand between the landmarks at metacarpale II and metacarpale V is measured with a sliding caliper."),
    ("palmlength", "PALM LENGTH",
     "The distance between the center of the crease at the base of the middle finger (digit III, base) and the ventral stylion landmark is measured with a Poech sliding caliper."),
    ("footlength", "FOOT LENGTH",
     "The maximum length of the right foot is measured with a Brannock Device(R)."),
    ("footbreadthhorizontal", "FOOT BREADTH, HORIZONTAL",
     "The maximum breadth of the right foot is measured with a Brannock Device(R)."),
    ("headcircumference", "HEAD CIRCUMFERENCE",
     "The maximum circumference of the head above the attachment of the ears is measured with a tape passing just above the ridges of the eyebrows and around the back of the head."),
    ("headlength", "HEAD LENGTH",
     "The distance from the glabella landmark to the opisthocranion landmark is measured with a spreading caliper."),
    ("headbreadth", "HEAD BREADTH",
     "The maximum horizontal breadth of the head above the plane of attachment of the ears is measured with a spreading caliper."),
    ("bizygomaticbreadth", "BIZYGOMATIC BREADTH",
     "The maximum horizontal breadth of the face (between the zygomatic arches) at the left and right zygion landmarks is measured with a spreading caliper."),
    ("interpupillarybreadth", "INTERPUPILLARY BREADTH",
     "The distance between the two pupils is measured with a pupillometer."),
    ("mentonsellionlength", "MENTON-SELLION LENGTH",
     "The distance between the menton landmark and the sellion landmark is measured with a sliding caliper."),
    ("tragiontopofhead", "TRAGION - TOP OF HEAD",
     "The vertical distance between the right tragion landmark and the horizontal plane tangent to the top of the head is measured with a beam caliper with a paddle blade."),
    ("earlength", "EAR LENGTH",
     "The length of the right ear, from its highest to lowest points on a line parallel to the long axis of the ear, is measured with a sliding caliper."),
]

# Variables of the correlation matrices: the ones the identity axes (height, mass, build, fat
# proxies, limb lengths, breadths, circumferences) are expected to drive.
CORRELATION_VARS = [
    "stature", "weightkg", "bmi", "age",
    "chestcircumference", "waistcircumference", "buttockcircumference", "thighcircumference",
    "lowerthighcircumference", "calfcircumference", "bicepscircumferenceflexed",
    "forearmcircumferenceflexed", "neckcircumference", "shouldercircumference",
    "biacromialbreadth", "bideltoidbreadth", "chestbreadth", "chestdepth", "waistbreadth",
    "waistdepth", "hipbreadth", "bicristalbreadth", "buttockdepth",
    "sittingheight", "span", "acromialheight", "trochanterionheight", "crotchheight",
    "lateralfemoralepicondyleheight", "lateralmalleolusheight",
    "acromionradialelength", "radialestylionlength", "buttockkneelength",
    "handlength", "handbreadth", "footlength", "footbreadthhorizontal",
    "headcircumference", "waist_to_height", "waist_to_hip",
]

DOD_RACE = {"1": "White", "2": "Black", "3": "Hispanic", "4": "Asian", "5": "Native American",
            "6": "Pacific Islander", "8": "Other"}

DEMOGRAPHIC = {"subjectid", "gender", "date", "installation", "component", "branch",
               "primarymos", "subjectsbirthlocation", "subjectnumericrace", "ethnicity",
               "dodrace", "age", "heightin", "weightlbs", "writingpreference"}


class InputError(Exception):
    """A CSV that does not have the expected shape or values."""


def read_csv(path, sex):
    """Reads one ANSUR II CSV; returns (columns, measurement dict col -> list, demographics)."""
    with open(path, "rb") as f:
        raw = f.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise InputError(f"{path}: larger than {MAX_CSV_BYTES} bytes")
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("latin-1")  # never fails; only numeric and ASCII columns are interpreted
    rows = csv.reader(io.StringIO(text, newline=""))
    header = next(rows, None)
    if header is None:
        raise InputError(f"{path}: empty")
    header = [h.strip().lower() for h in header]
    if len(set(header)) != len(header):
        raise InputError(f"{path}: duplicate column names")
    measures = [h for h in header if h not in DEMOGRAPHIC]
    if len(measures) != 93:
        raise InputError(f"{path}: expected 93 measurement columns, found {len(measures)}")
    idx = {h: i for i, h in enumerate(header)}
    data = {m: [] for m in measures}
    demo = {"age": [], "dodrace": [], "component": [], "branch": [], "date": []}
    count = 0
    for line_no, row in enumerate(rows, start=2):
        if not row:
            continue
        count += 1
        if count > MAX_ROWS:
            raise InputError(f"{path}: more than {MAX_ROWS} rows")
        if len(row) != len(header):
            raise InputError(f"{path}:{line_no}: {len(row)} fields, header has {len(header)}")
        if row[idx["gender"]].strip() != sex:
            raise InputError(f"{path}:{line_no}: Gender is not {sex}")
        for m in measures:
            v = parse_int(row[idx[m]], VALUE_RANGE, f"{path}:{line_no}:{m}")
            data[m].append(v / 10.0 if m in TENTHS else float(v))
        demo["age"].append(float(parse_int(row[idx["age"]], AGE_RANGE, f"{path}:{line_no}:age")))
        for k in ("dodrace", "component", "branch", "date"):
            demo[k].append(row[idx[k]].strip()[:64])
    if count == 0:
        raise InputError(f"{path}: no data rows")
    return measures, data, demo, digest, count


def parse_int(s, rng, where):
    s = s.strip()
    if not s.isdigit() or len(s) > 6:
        raise InputError(f"{where}: not a small decimal integer: {s[:20]!r}")
    v = int(s)
    if not rng[0] <= v <= rng[1]:
        raise InputError(f"{where}: {v} outside {rng}")
    return v


def percentile(sorted_xs, p):
    """Hyndman-Fan type 7 (linear interpolation between order statistics)."""
    h = (len(sorted_xs) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])


def summary(xs, digits):
    n = len(xs)
    mean = math.fsum(xs) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    s = sorted(xs)
    r = lambda v: round(v, digits)  # noqa: E731
    return {"n": n, "mean": r(mean), "sd": r(sd), "min": r(s[0]), "p5": r(percentile(s, 0.05)),
            "p50": r(percentile(s, 0.50)), "p95": r(percentile(s, 0.95)), "max": r(s[-1])}


def pearson(a, b):
    n = len(a)
    ma = math.fsum(a) / n
    mb = math.fsum(b) / n
    sab = math.fsum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = math.fsum((x - ma) ** 2 for x in a)
    sbb = math.fsum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


def derive(data, demo):
    """Per-subject derived variables: BMI, waist/height, waist/hip (waist at omphalion), age."""
    st = data["stature"]
    data["bmi"] = [w / (h / 1000.0) ** 2 for w, h in zip(data["weightkg"], st)]
    data["waist_to_height"] = [w / h for w, h in zip(data["waistcircumference"], st)]
    data["waist_to_hip"] = [w / h for w, h in zip(data["waistcircumference"],
                                                  data["buttockcircumference"])]
    data["age"] = demo["age"]


def digits_for(var):
    if var in ("waist_to_height", "waist_to_hip"):
        return 4
    if var in ("weightkg", "bmi", "age"):
        return 2
    return 1


def population(demo, n):
    def counts(key, names=None):
        out = {}
        for v in demo[key]:
            label = names.get(v, f"code {v}") if names else v
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
    return {"n": n, "dod_race": counts("dodrace", DOD_RACE), "component": counts("component"),
            "branch": counts("branch"), "age_years": summary(demo["age"], 2)}


def main(argv):
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    male_path, female_path, retrieved, out_path = argv[1:]
    sexes = {}
    for sex, path in (("male", male_path), ("female", female_path)):
        measures, data, demo, digest, n = read_csv(path, sex.capitalize())
        derive(data, demo)
        sexes[sex] = (measures, data, demo, digest, n)
    if sexes["male"][0] != sexes["female"][0]:
        raise InputError("the two files do not have the same measurement columns")
    ids = sexes["male"][0] + ["bmi", "waist_to_height", "waist_to_hip", "age"]
    stats = {s: {v: summary(sexes[s][1][v], digits_for(v)) for v in ids} for s in sexes}
    midpoint = {v: round((stats["male"][v]["mean"] + stats["female"][v]["mean"]) / 2,
                         digits_for(v)) for v in ids}
    corr = {"variables": CORRELATION_VARS}
    for s in sexes:
        d = sexes[s][1]
        corr[s] = [[round(pearson(d[a], d[b]), 3) for b in CORRELATION_VARS]
                   for a in CORRELATION_VARS]
    units = {v: ("kg" if v == "weightkg" else "kg/m^2" if v == "bmi" else "years" if v == "age"
                 else "1" if v in ("waist_to_height", "waist_to_hip") else "mm") for v in ids}
    out = {
        "format": {"name": "humanart-anthropometry-summary", "version": 1, "note": FORMAT_NOTE},
        "source": {
            "dataset": "ANSUR II (2012 US Army Anthropometric Survey) public working databases",
            "publisher": "US Army Natick Soldier Research, Development and Engineering Center "
                         "(NSRDEC); files hosted by the OPEN Design Lab, Penn State",
            "page_url": PAGE_URL,
            "male_csv_url": MALE_URL,
            "female_csv_url": FEMALE_URL,
            "memorandum_url": MFR_URL,
            "report_url": REPORT_URL,
            "report": "Gordon C.C., Blackwell C.L., Bradtmiller B., Parham J.L., Barrientos P., "
                      "Paquette S.P., Corner B.D., Carson J.M., Venezia J.C., Rockwell B.M., "
                      "Mucher M., Kristensen S. (2014). 2012 Anthropometric Survey of U.S. Army "
                      "Personnel: Methods and Summary Statistics. NATICK/TR-15/007.",
            "retrieved": retrieved,
            "male_csv_sha256": sexes["male"][3],
            "female_csv_sha256": sexes["female"][3],
            "licence": "Released by the US Army for unlimited public distribution. Memorandum for "
                       "Record RDNS-SEW-THA, 14 April 2017: 'These databases and this memorandum "
                       "have been reviewed and cleared for UNLIMITED PUBLIC RELEASE.' Report "
                       "NATICK/TR-15/007, distribution statement: 'Approved for public release; "
                       "distribution is unlimited.' Neither document uses the words 'public "
                       "domain'; no other licence terms are attached on the hosting page.",
            "population_note": "US Army soldiers (Regular Army, Army Reserve, Army National "
                               "Guard) measured Oct 2010 - Apr 2012; the public sample was drawn "
                               "by stratified random sampling to match Army demographics of "
                               "30 Sept 2011 (TR-15/007). A working-age, physically active "
                               "military population, not a general-population sample: the "
                               "memorandum says it 'may or may not be representative of other "
                               "populations of interest' (age and race make-up under "
                               "'population'). The memorandum warns that combining "
                               "the male and female databases 'will result in a sample that is "
                               "not representative of any real population'; sex_midpoint_mean "
                               "is a design midpoint, not a statistic of anyone. Self-reported "
                               "Heightin and Weightlbs are not used.",
            "unit_fixes": "weightkg divided by 10 (stored as kg*10, documented); "
                          "interpupillarybreadth divided by 10 (stored in 0.1 mm, undocumented, "
                          "established by comparison with the report's published percentiles)",
            "generator": "tools/anthropometry/ansur2_summary.py",
        },
        "units": units,
        "variables": [{"id": c, "title": t, "definition": d, "unit": units.get(c, "mm")}
                      for c, t, d in DEFINITIONS],
        "derived": {
            "bmi": "weightkg / (stature in m)^2",
            "waist_to_height": "waistcircumference / stature (waist at omphalion, the navel, not "
                               "the WHO site midway between lowest rib and iliac crest)",
            "waist_to_hip": "waistcircumference / buttockcircumference (same caveat)",
            "age": "Age column, years",
        },
        "stats": stats,
        "sex_midpoint_mean": midpoint,
        "correlation": corr,
        "population": {s: population(sexes[s][2], sexes[s][4]) for s in sexes},
    }
    for c, _, _ in DEFINITIONS:
        if c not in stats["male"]:
            raise InputError(f"definition for unknown column {c}")
    for c in CORRELATION_VARS:
        if c not in stats["male"]:
            raise InputError(f"correlation variable {c} unknown")
    text = json.dumps(out, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(text)} bytes, male n={sexes['male'][4]}, "
          f"female n={sexes['female'][4]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
