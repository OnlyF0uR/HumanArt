# Data provenance

Every asset in the repository with its source, licence and, for captured data, the consent record id
(`[LIC-01]`, `[GOAL-06]`). Nothing is added to `data/` without a row here.

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/fonts/Inter-Regular.ttf`, `Inter-SemiBold.ttf`, `Inter-Italic.ttf` | Inter, The Inter Project Authors (https://github.com/rsms/inter); copied from the reference application's `crates/desktop/assets/` | SIL Open Font License 1.1 (`data/fonts/Inter-LICENSE.txt`) | not applicable | UI sans font `[LIC-04]`; source for the baked MSDF atlas |
| `data/fonts/JetBrainsMono-Regular.ttf` | JetBrains Mono, The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono); copied from the reference application's assets | SIL Open Font License 1.1 (`data/fonts/JetBrainsMono-LICENSE.txt`) | not applicable | Monospaced font for the console and numeric fields `[LIC-04]` |
| `data/fonts/Lucide.ttf` | Lucide icon font, Lucide Icons and Contributors (https://lucide.dev); copied from the reference application's assets | ISC (`data/fonts/Lucide-LICENSE.txt`), a permissive licence equivalent in effect to the OFL for bundling `[LIC-04]` | not applicable | UI icons; source for the baked icon atlas |

### Test fixtures and dev-time scripts (WP-03)

Generated in-house unless a row says otherwise; the tools ran at development time only `[DEP-04]`,
never from the build or the tests. "Project" means the files carry the repository's own terms and
contain no third-party content.

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `crates/io/tests/fixtures/deflate/make_fixtures.py` | Written in-house (Python 3 standard library) | project | not applicable | Dev-time generator of the DEFLATE / zlib fixtures below |
| `crates/io/tests/fixtures/deflate/input.bin` | Generated from a fixed-seed LCG by `make_fixtures.py` | project | not applicable | 70 004 bytes; the reference content the zlib fixtures decode to |
| `crates/io/tests/fixtures/deflate/zlib_l0.zz`, `zlib_l1.zz`, `zlib_l6.zz`, `zlib_l9.zz`, `zlib_w9.zz`, `zlib_fixed.zz`, `zlib_huffman.zz`, `zlib_rle.zz`, `zlib_dict.zz`, `zeros_4m.zz`, `raw_l6.deflate` | `input.bin` (or zeros, or a 22-byte string) compressed by Python's zlib (1.3.1.zlib-ng) through `make_fixtures.py` | project | not applicable | Stored, fixed, dynamic, Huffman-only, RLE, 512-byte window, preset dictionary, bomb-shaped and raw streams |
| `crates/io/tests/fixtures/png/make_fixtures.py` | Written in-house: a PNG writer independent of the Rust code (Python 3 standard library) | project | not applicable | Also drives the dev-time ffmpeg cross-check (`--check`) and the `ff_*` fixtures (`--ffmpeg`) |
| `crates/io/tests/fixtures/png/c*.png`/`.bin`, `rgba8_i_*`, `grey1_i_3x5`, `srgb`, `gama_22`, `gama_linear`, `iccp_over_gama`, `split_idat` (48 pairs) | Pixels from a fixed-seed LCG, written by `make_fixtures.py` with Python's zlib; `.bin` is the expected decode | project | not applicable | The iCCP payload of `iccp_over_gama.png` is placeholder bytes, not a real ICC profile |
| `crates/io/tests/fixtures/png/ff_rgb24`, `ff_rgba64`, `ff_gray16`, `ff_pal8`, `ff_monob`, `ff_ya8` (`.png`/`.bin`, 6 pairs) | ffmpeg 9.0.2 (gyan.dev build) `testsrc2` generated pattern, encoded to PNG and decoded to raw samples by ffmpeg | project (generated pattern, no third-party image) | not applicable | Second, independent encoder for the decoder tests |
| `crates/io/tests/fixtures/hdr/ffmpeg_24x17.hdr` | Synthetic picture (fixed LCG) from `crates/io/tests/fixtures/exr/gen_src.py`, encoded by ffmpeg 9.0.2's `hdr` encoder | project | not applicable | Regenerate with `crates/io/tests/fixtures/exr/gen_src.py`; `xcheck.py` beside it compares with ffmpeg |
| `crates/io/tests/fixtures/hdr/ffmpeg_24x17.gbrpf32le.bin` | The `.hdr` above decoded by ffmpeg 9.0.2 to planar G, B, R f32 LE | project | not applicable | Expected decode |
| `crates/io/tests/fixtures/exr/ffmpeg_rgb_half_zip.exr`, `ffmpeg_rgb_half_zips.exr`, `ffmpeg_rgba_float_zips.exr`, `ffmpeg_rgba_float_rle.exr`, `ffmpeg_rgb_float_none.exr`, `ffmpeg_y_half_rle.exr` | 37x21 synthetic picture from `crates/io/tests/fixtures/exr/gen_src.py` (fixed LCG; gbr / gbra / gray planes) encoded by ffmpeg 9.0.2's `exr` encoder (`-compression zip16/zip1/rle/none -format half/float`) | project | not applicable | No PIZ fixture: ffmpeg cannot write PIZ (see `docs/blockers.md` B-005). ffmpeg truncates rather than rounds float to half |
| `crates/io/tests/fixtures/exr/*.gbrpf16le.bin`, `*.gbrpf32le.bin`, `*.gbrapf32le.bin`, `*.grayf16le.bin` (6 files) | Each `.exr` above decoded by ffmpeg 9.0.2 to its decoder's native planar format (`-c:v rawvideo -pix_fmt ...`) | project | not applicable | Native f16 output used for half files (ffmpeg's f16 to f32 path is inexact) |
| `tools/ttf_crosscheck.py` | Written in-house (Python 3 standard library, `struct`); shares no code with the Rust parser | project | not applicable | Dev-time cross-check of the TrueType parser |
| `crates/io/tests/fixtures/ttf/Inter-Regular.txt`, `Inter-SemiBold.txt`, `Inter-Italic.txt`, `JetBrainsMono-Regular.txt`, `Lucide.txt` | Output of `tools/ttf_crosscheck.py --fixtures` run on the fonts in `data/fonts/` (rows above) | project (facts read from the OFL / ISC fonts) | not applicable | Expected metrics, mappings, outlines and kerning |
| `crates/io/tests/fixtures/obj/cube.obj`, `cube.mtl` | Written by hand in-house | project | not applicable | OBJ / MTL reader and writer tests |
| `crates/io/tests/fixtures/ply/make_fixtures.py` | Written in-house (Python 3 standard library, `struct`), independent of the Rust writer | project | not applicable | Dev-time generator |
| `crates/io/tests/fixtures/ply/tetra_ascii.ply`, `tetra_le.ply`, `tetra_be.ply` | One tetrahedron with every PLY scalar type, a list and three elements, written by `make_fixtures.py` | project | not applicable | The Rust writer reproduces all three byte for byte |
| `docs/reference-conventions.md` (the repository's own document), read by the `crates/io/src/lz/mod.rs` tests through `include_bytes!` | This repository | project | not applicable | Sample English text for the LZ round-trip tests; no separate fixture file |
| `crates/io/tests/fixtures/jpeg/make_fixtures.py` | Written in-house (Python 3 standard library): a JPEG writer independent of the Rust code, which also drives ffmpeg 9.0.2 (gyan.dev full build) | project | not applicable | Dev-time generator; `--bench DIR` writes the large timing files (not in the repository) |
| `crates/io/tests/fixtures/jpeg/py_*.jpg`, `py_*.bin` (19 pairs, plus `py_444_rgb.bin`) | Fixed-seed pattern encoded by `make_fixtures.py`'s own writer; each `.bin` is ffmpeg 9.0.2's decoding of one-component baseline twins holding exactly that component's quantised coefficients | project (generated pattern, no third-party image; the io-jpeg agent proposed CC0) | not applicable | Baseline, progressive, restart, 4:1:1 to 4:4:4, grey, RGB, CMYK, YCCK, EXIF orientation fixtures |
| `crates/io/tests/fixtures/jpeg/ff_420`, `ff_422`, `ff_444` (`.jpg`/`.bin`), `ff_444_rgb.bin` | The same pattern encoded by ffmpeg 9.0.2's MJPEG encoder and decoded by ffmpeg to planar samples (`_rgb.bin`: ffmpeg's RGB24 conversion) | project (as above) | not applicable | Second, independent encoder for the decoder tests |
| `crates/io/tests/fixtures/gltf/make_fixtures.py` | Written in-house (Python 3 standard library), independent of the Rust reader and writer | project | not applicable | Dev-time generator of the glTF fixtures below |
| `crates/io/tests/fixtures/gltf/triangle.gltf`, `sparse_morph.gltf`, `sparse morph.bin`, `skinned.glb`, `quantized.gltf`, `*.expect.json` (4) | Written by `make_fixtures.py`; `.expect.json` holds the values the reader must decode | project | not applicable | They embed `png/rgba8_i_1x1.png` and `png/c2_d8.png` from the PNG fixtures (rows above). No KTX2 fixture file: the KTX2 tests build their files in code |

### Parameter registry and document fixtures (WP-07)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/template/params/identity.json`, `face.json`, `body.json`, `expression.json`, `skin.json`, `eyes.json`, `mouth.json`, `hair.json` (224 parameters, 82 of them expression, at the doc agent's run) | Authored in-house from the original specification (removed 2026-11-19), sections 4, 6, 7 and Appendix B; values from the spec (expression 0..1, cornea IOR 1.376, aqueous IOR 1.336) or marked `provisional` with the reason in each entry | project | not applicable | No external data. Shape-axis ranges and most appearance defaults are provisional; no `lighting_camera` parameters |
| The 45 `skin.features.*` entries in `data/template/params/skin.json` (round 5) | Copied from `GeneratorKind::params` (`crates/model/src/skin_appearance/features.rs`) by the skin-finish agent's script `target/agents/skin-finish/scripts/gen_features_json.py` (outside the repository) | project | not applicable | Artistic defaults of the 11 `[OTH-02]` generators; `features::tests::oth02_every_generator_parameter_is_in_the_registry` keeps both in step. The identity agent's round two is adding more entries to `data/template/params/` (rows when it finishes) |
| The 10 breast, nipple, areola, waist, glute, hip-flank and thigh entries in `data/template/params/body.json` (`body.torso.bust_size`, `breast_projection`, `breast_ptosis`, `breast_spacing`, `nipple_size`, `areola_size`, `waist_width`, `glute_size`, `body.legs.hip_flank`, `thigh_fullness`; round 6) | Added by the identity agent (round two) for its components (`identity::BINDINGS`, test `reg_01_bindings_match_the_registry_files`) | project | not applicable | Authored axes; of these only waist width, glute size and thigh fullness are calibrated to ANSUR II (ADR-051) |
| `crates/doc/tests/fixtures/document-v1.anthropos` | Written by `humanart-doc` itself: canonical text of `document::tests::sample` (synthetic content) | project | not applicable | Format version 1 fixture; the test checks it byte for byte |

### UI atlases and goldens (WP-12)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/fonts/regular.anfont`, `semibold.anfont`, `italic.anfont` | Baked in-house by `tools/fontbake/bake.sh` (`humanart_ui::fontbake`, MSDF) from the Inter TTFs (rows above); Latin set, 32 px/em, range 4 | SIL Open Font License 1.1 (derived from Inter, `data/fonts/Inter-LICENSE.txt`) | not applicable | UI text atlases `[UI-03]` |
| `data/fonts/mono.anfont` | Same tool from `JetBrainsMono-Regular.ttf` | SIL Open Font License 1.1 (`data/fonts/JetBrainsMono-LICENSE.txt`) | not applicable | Monospaced text atlas |
| `data/fonts/icons.anfont` | Same tool from `Lucide.ttf`, all mapped icons (2071 per the ui-core agent; 2072 glyphs with the fallback), 24 px/em, range 3 | ISC (`data/fonts/Lucide-LICENSE.txt`) | not applicable | Icon atlas |
| `crates/ui/tests/golden/painter_*.png`, `text_specimen_*.png`, `widgets_*.png` | Rendered in-house by the crate's own tests on PC-1's NVIDIA GPU (the first run writes them, `HUMANART_BLESS=1` rewrites), checked by eye by the agent | project | not applicable | Golden screenshots `[AC-WP12-3]`; not yet signed off by a human (B-002). A `*.actual.png` beside them is a failed run's output, not a golden |
| `crates/ui/tests/golden/controls_*.png` (incl. `controls_75`, `controls_high_contrast`), `popups_*.png`, `editors_*.png`, `views_*.png`, `dock_*.png`, `bench_scene.png`, `text_area.png` | Rendered in-house by the crate's own tests on PC-1's NVIDIA GPU (2/255 tolerance), checked by eye by the ui-widgets agent | project | not applicable | Golden screenshots `[AC-WP12-2]`; not signed off by a human (B-002); other GPUs may need re-blessing (B-003) |

### Data tables in source code (WP-02)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `crates/math/src/color/checker.rs` `REFERENCE_LAB_D50` (ColorChecker Classic, 24 patches, CIELAB D50/2°, post-November-2014 formulation) | X-Rite (2016), "New color specifications for ColorChecker SG and Classic Charts", as reproduced in colour-science `chromaticity_coordinates.py` `DATA_COLORCHECKER24_AFTER_NOV2014_CIE_LAB` (GitHub `colour-science/colour`, develop branch, fetched 2026-11-19) | Published reference values (facts); colour-science is BSD-3-Clause | not applicable | Copied unchanged, 2 decimals `[CMP-02]` |
| CIEDE2000 test pairs (34) in the `crates/math/src/color/delta_e.rs` tests | G. Sharma, W. Wu, E. N. Dalal (2005), "The CIEDE2000 color-difference formula: implementation notes, supplementary test data, and mathematical observations", Table 1 | Published test data | not applicable | Test-only; evidence for `[AC-WP02-1]` |
| ACES fitted tone-mapping constants in `crates/math/src/color/tonemap.rs` | S. Hill, BakingLab `ACES.hlsl` | MIT | not applicable | Constants only |
| Khronos PBR Neutral constants in `crates/math/src/color/tonemap.rs` | Khronos PBR Neutral tone mapper specification (2024) | Apache-2.0 (sample code) | not applicable | Start of compression 0.76, desaturation 0.15 |
| ASTM D1535 Munsell Value polynomial in `crates/math/src/color/munsell.rs` | ASTM D1535 | Published standard formula | not applicable | 5 coefficients; Value only (hue and chroma need the renotation data, `docs/blockers.md` B-004) |
| Other published colour-science constants in `crates/math/src/color/` (Oklab matrices, Bradford matrix, CIE daylight locus and Planckian approximations, McCamy CCT, ITA category limits) | Ottosson (2020); Lam (1985); CIE 15; Kang et al. (2002) and Krystek (1985) approximations; McCamy (1992); Chardon (1991) / Del Bino (2006), as cited in each module's doc comments | Published formulas (facts) | not applicable | Coefficients only, no tabulated data sets |

### Spectral tables and published models in source code (WP-13)

All in `crates/model/src/skin_appearance/spectra_table.rs` (81 bands, 380-780 nm at 5 nm), generated by
`tools/spectra/gen_spectra.py` (dev-time, Python stdlib) from the downloads its docstring lists. The
downloaded source files are not kept in the repository; the script names each source and its DOI or URL.

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| CIE 1931 2° colour-matching functions x̄, ȳ, z̄, 380-780 nm at 5 nm | CIE dataset, DOI 10.25039/CIE.DS.xvudnb9b (`files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv`), fetched 2026-11-19 | **CC BY-SA 4.0**: the embedded 81-row subset is attributed and shared under the same licence (owner decision, ADR-038) | not applicable | Sampled from the 1 nm table at the bands, values unchanged |
| CIE standard illuminant D65, 380-780 nm at 5 nm | CIE dataset, DOI 10.25039/CIE.DS.hjfjmt59 (`files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv`), fetched 2026-11-19 | **CC BY-SA 4.0** (ADR-038) | not applicable | Values unchanged |
| Oxy- and deoxy-haemoglobin molar extinction coefficients (cm⁻¹/M) | S. Prahl, OMLC, `omlc.org/spectra/hemoglobin/summary.html` (compiled from Gratzer and Kollias), fetched 2026-11-19 | No licence stated; attributed (B-008 notes it) | not applicable | 2 nm table; odd wavelengths are the mean of their neighbours |
| β-carotene molar extinction (in hexane) | PhotochemCAD 2.1a (Du 1998; Dixon 2005) via S. Prahl, `omlc.org/spectra/PhotochemCAD/data/041-abs.txt`, fetched 2026-11-19 | No licence stated; attributed (B-008 notes it) | not applicable | Linearly interpolated; negative noise clamped to 0; 0 beyond the table's 700 nm end |
| Melanin, baseline and scattering laws in `crates/model/src/skin_appearance/pigment.rs` | Donner & Jensen 2006 (EGSR) eqs. 3-5 and 8; Jacques 1998, "Skin Optics" (`omlc.org/news/jan98/skinoptics.html`) | Published formulas (facts) | not applicable | Coefficients only; the quoted values are checked in tests; derivation in `docs/shader-notes.md` "Skin pigment model" |
| Diffusion `Rd` / `Fdr`, micro-shadow, PCG hash, histogram-preserving blending, `erf` approximation in `skin_appearance` and `shaders/skin/` | Jensen & Buhler 2002; Jensen et al. 2001; Chan 2018; Jarzynski & Olano 2020; Heitz & Neyret 2018; Abramowitz & Stegun 7.1.26 | Published formulas (facts) | not applicable | Constants only |
| Melanin packaging (`packaged_absorption`) and the fitted constants `Mapping::eumelanin_packet_mm` 0.035 mm and `melanin_max` 1.0 in `crates/model/src/skin_appearance/pigment.rs` (round 5) | Duysens 1956, Biochim. Biophys. Acta 19:1-12 (flattening formula); constants fitted by the skin-calib agent to the ISSA summary below (ADR-047) | Published formula (facts); fitted values project | not applicable | Effective optical parameters, not a melanosome size |
| Blood packaging per haemoglobin species and the fitted constant `Mapping::blood_vessel_radius_mm` 0.013 mm in `crates/model/src/skin_appearance/pigment.rs` (`Mapping::DEFAULT`, round 6) | Vessel correction factor of Verkruysse et al. 1997, Phys. Med. Biol. 42:51-65, in the form of van Veen, Verkruysse & Sterenborg 2002, Opt. Lett. 27:246-248 (published formula); radius fitted by the skin-wire agent to the ISSA summary below (ADR-053) | Published formula (facts); fitted value project | not applicable | Effective parameter of the fit, not a measured vessel size |

### Skin reference swatches (WP-13, ADR-047)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/test-scenes/skin-reference.json` | Aggregates of the International Skin Spectra Archive: Lu Y. et al. 2025, "The International Skin Spectra Archive (ISSA): a multicultural human skin phenotype and colour spectra collection", Scientific Data 12, 487, doi:10.1038/s41597-025-04857-5; data figshare doi:10.6084/m9.figshare.28228571.v4, file `ISSA_17_Jan_2025_Yan_Lu.xlsx` (9 320 520 bytes, SHA-256 7396aa76...1161, MD5 matching figshare's v4 listing), fetched 2026-11-19 | CC BY 4.0 (authors, attribution and changes recorded in the file) | not applicable (published, de-identified dataset; only aggregates stored, no individual records) | 15 256 specular-included spectrophotometer records of 2 107 subjects in 8 self-reported groups; Lab (CIE 1931 2°, D65) statistics and mean spectra 400-700 nm by 6 ITA categories, 10 L* bands and 12 body sites. No Fitzpatrick phototypes in the data (ITA stands in, an open owner question, ADR-047). Generated by the script below; the lead's rebuild was byte-identical |
| `tools/skin-reference/gen_skin_reference.py` | Written in this repository by the skin-calib agent (Python 3 standard library; run with `python -I`, workbook path, retrieval date and output as arguments) | project | not applicable | Dev-time generator `[DEP-04]`, not used by the build or tests; size, zip-ratio, DOCTYPE, row and column limits, every record range-checked; deterministic |

### Anthropometry (ADR-046)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/anthropometry/ansur2.json` | Summary statistics computed by `tools/anthropometry/ansur2_summary.py` from the ANSUR II public working databases (US Army NSRDEC, survey 2010-2012; files from https://www.openlab.psu.edu/ansur2/, retrieved 2026-11-19; SHA-256 of both CSVs recorded in the file); variable titles and one-sentence definitions from Gordon et al. 2014, NATICK/TR-15/007, with attribution | US Army release, in the publisher's words: "cleared for UNLIMITED PUBLIC RELEASE" (Memorandum for Record RDNS-SEW-THA, 14 April 2017) and "Approved for public release; distribution is unlimited" (TR-15/007). Neither says "public domain" | not applicable (de-identified public release; only aggregates stored, no per-subject rows) | US Army soldiers 2010-2012, age 17-58, 4082 M / 1986 F: military population bias recorded in the file (ADR-046). Percentiles checked against the report's tables (max difference 0.5 mm). `interpupillarybreadth` is stored in 0.1 mm in the CSV and divided by 10 (`source.unit_fixes`). The lead's rebuild from the CSVs was byte-identical |
| `tools/anthropometry/ansur2_summary.py` | Written in this repository by the anatomy-data agent (Python 3 standard library; CSV paths as arguments) | project | not applicable | Dev-time generator `[DEP-04]`, not used by the build or tests; rejects oversized files, wrong field counts, out-of-range values and duplicate or missing columns |

### Template (WP-06, ADR-006, ADR-039)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/template/landmarks.json` | Generated by `humanart_model::generate::landmarks` (test `mod_06_landmarks_are_valid_symmetric_and_documented`; `HUMANART_BLESS=1` rewrites it, otherwise the test compares it byte for byte) | project (procedural, `[LIC-02]` (c)) | not applicable | 173 face + 68 body landmarks with Appendix C names; re-blessed several times in rounds 5-6 (positions only, no name added or changed) |
| Generated template (`generate::build::template`; not checked in, the package is written to `target/agents/template-gen/template.antmpl`) | Procedural, `humanart_model::generate` (template-gen agent); identity and expression components added by `identity::with_identity` and `expression::build::with_expressions`; the build adds its own provenance rows to the package | project (procedural, `[LIC-02]` (c)) | not applicable | Body proportions are ANSUR II sex midpoints read from `data/anthropometry/ansur2.json`, cited per constant in `generate/anatomy.rs` and tested (`anat_01_constants_match_the_ansur_ii_file`, `anat_02_surface_meets_the_ansur_constants`); head height, eye position and radius, jaw pivot, pelvis and spine joints, joint x offsets, finger and thumb chains and rest angles are unsourced design values marked in `anatomy.rs`; the surface-anatomy relief (`generate/relief.rs`), skin weights (ADR-049) and identity amplitudes outside the calibrated axes (ADR-051) are authored, not measured |

### Expression data (WP-19)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| `data/template/expressions/rig.json`, `drivers.json`, `correctives.json`, `presets.json`, `wrinkles.json` | Authored in-house by the expression agent (dev-script output, then kept as data) | project | not applicable | Every amount, angle, gain and strength is a model choice, not a measurement. `presets.json`: the emotion action-unit sets follow the prototype AU combinations associated with the six basic emotions in the FACS / EMFACS literature (Ekman and Friesen), recalled from memory by the agent and not checked against the manual, with authored weights; the visemes are the common 15-viseme set. `rig.json` adds 7 `viseme.*` controls the registry lacks (ADR-043) |
| Expression targets in the built template (`humanart_model::expression::build::with_expressions`) | Procedural, `humanart_model::expression::recipes` (ADR-043) | project | not applicable | No data file; the build adds its own provenance row to the template it returns |

### Published models in source code (WP-14, WP-17)

| Asset | Source | Licence | Consent record | Notes |
|---|---|---|---|---|
| Six-Gaussian skin diffusion profile (variances and RGB weights) in `crates/render/src/sss.rs` (`PROFILE_*`) and `shaders/render/subsurface.wgsl` | d'Eon & Luebke 2007, "Advanced Techniques for Realistic Real-Time Skin Rendering", GPU Gems 3 ch. 14, table 14-1 (fit by d'Eon, Luebke and Enderton to the three-layer skin model) | Published values (facts) | not applicable | Constants only (ADR-044) |
| Separable SSS approximation, back-plane transmittance, shadow-map thickness offset, wrap lighting; pre-integrated skin table | Jimenez et al. 2010, "Real-Time Realistic Skin Translucency"; Jimenez et al. 2015, "Separable Subsurface Scattering"; Penner 2011, "Pre-Integrated Skin Shading" | Published formulas (facts) | not applicable | As cited in the module docs of `sss.rs` and `subsurface.wgsl` |
| Melanin absorption in `crates/model/src/hair/color.rs`: eumelanin (0.419, 0.697, 1.37), pheomelanin (0.187, 0.4, 1.05) | pbrt-v3 `src/materials/hair.cpp`, `HairBSDF::SigmaAFromConcentration`; the same values in pbrt-v4 (Pharr, Jakob, Humphreys 2023) sec. 9.9.8 | pbrt-v3 source is BSD-2-Clause; values are published facts | not applicable | Constants only |
| Hair melanin presets: black 8, brown 1.3, blonde 0.3 eumelanin | pbrt-v4 book sec. 9.9.8 "Hair Absorption Coefficients" | Published values (facts) | not applicable | `hair::color::PRESETS` |
| Default hair colour and roughness: eumelanin 1.3, pheomelanin 0, `beta_n` 0.3 | pbrt-v3 `CreateHairMaterial` defaults | BSD-2-Clause (pbrt-v3) | not applicable | Defaults of the `GroomParams` keys `melanin.*`, `shading.azimuthal_roughness` |
| Dye colour to `sigma_a` fit: 5.969, 0.215, 2.532, 10.73, 5.574, 0.245 | Chiang, Bitterli, Tappan, Burley, "A Practical and Controllable Hair and Fur Model for Production Path Tracing", CGF 35(2) 2016; constants as in pbrt-v3 `HairBSDF::SigmaAFromReflectance` | Published formula (facts); pbrt-v3 BSD-2-Clause | not applicable | `hair::color::sigma_a_from_reflectance` |

Hair preset lengths, curl values, brow / lash / beard shapes and card counts
(`humanart_model::hair::{presets, features, facial, cards}`) are styling choices of the hair agent,
not measurements.

### Origin unknown

Files in the repository whose origin no agent recorded. None among the files of the finished WP-02,
WP-03, WP-04, WP-05, WP-06 (data model, LOD generator), WP-07, WP-12 (core), WP-28 (texture half)
and WP-31 work (checked 2026-11-19 by integrator2 against every file under `data/` and
`crates/*/tests/`). The WP-05 and WP-28 work added no data files: the WGSL under `shaders/` is source
code, and the BC7/BC6H and mip tests generate their content from fixed seeds.

Round 3 (integrator3, 2026-11-19, from the agents' STATUS files and a listing of the non-source files
under `crates/` and `data/`): WP-08 (commands), WP-09 (CPU evaluation), WP-10 (GPU evaluation),
WP-11 (render) and WP-15 (bakers) added no data files; their tests generate their content in code
(the render review shots and the baker images are written to `target/agents/*/`, not to the
repository). WP-12 (widgets) added the goldens listed above, WP-13 (skin) the spectral table above.

Round 4 (integrator4, 2026-11-19, from the agents' STATUS files and a listing of the non-source
files under `crates/`, `shaders/`, `tools/` and `data/`): WP-14 (skin shading), WP-17 (hair),
WP-21 (application shell) and the fixes round added no data files (stills, previews, stress logs
and offscreen shots are written to `target/agents/*/`); WP-19 added the expression data above.

Round 5 (integrator5, 2026-11-19, from the agents' STATUS files and a listing of `data/` and
`tools/`): the LOD fix (WP-28) and the identity components (round one) added no data files (the
identity deltas are built in code; its previews are in `target/agents/identity/previews/`);
skin-finish added the `skin.features.*` registry entries and WGSL sources (`shaders/skin/wrinkle.wgsl`,
`wrinkle_drivers.wgsl`, `page_copy.wgsl`), its zone previews are in `target/agents/skin-finish/`;
anatomy-data and skin-calib added the anthropometry and skin-reference rows above. The downloads
(ANSUR II CSVs and PDFs, the ISSA workbook) stay in `target/agents/*/dl/`, outside the repository.

Round 6 (integrator6, 2026-11-19, from the agents' STATUS files and a listing of `data/`):
template-gen's rows are in the Template section above (`data/template/landmarks.json`, the generated
template; template-gen's round 3 is running, so later changes are not covered); identity round two
added 10 entries to `data/template/params/body.json` (row above); skin-wire added
the blood-packaging row above; skin-render (WGSL sources `shaders/render/skin_surface.wgsl` and
`detail_mips.wgsl`) and perf-gaps added no data files. Stills and previews are in
`target/agents/*/`, outside the repository.

### Reference material outside the repository

Visual targets used by agents and the lead that are not in the repository and are not shipped.

| Material | Origin | Use | Notes |
|---|---|---|---|
| `target/reference/char-01.png` | AI-generated by the owner (tool not named); owner statement 2026-11-19 | Visual target only (e.g. the identity agent's `preview_reference_comparison` and its side-by-side previews) | Not in the repository and not shipped; used to judge renders by eye, not as data |

Captured data: none (see `docs/blockers.md` B-001), so no consent
records exist.
