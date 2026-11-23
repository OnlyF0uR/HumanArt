# To do

What HumanArt still needs, most important first. This file replaces the original specification as the
plan. `PROGRESS.md` records what is done and the evidence for it, `docs/decisions.md` the decisions and
`docs/blockers.md` what is blocked. Requirement IDs such as `[EYE-05]` in code and tests are labels from
the original specification; `PROGRESS.md` keeps the evidence under them.

## Where we stand (2026-11-21)

The engineering base is broad and tested: codecs, GPU layer, shader system, document and commands,
CPU and GPU evaluation, render graph, UI toolkit, skin model calibrated against measured skin colour,
subsurface shading, bakers, LODs, texture compression. Since 2026-11-20 the eyes, mouth interior and
hair are drawn on the head in CLI renders and the editor, the body measurements match ANSUR II, and
the default face is the mean of 8 scanned people. The character still reads as **computer-made**: the
skin is waxy, the lashes and teeth look artificial and expressions read weakly. Data: HSRD-100 (face shape, skin detail) and BodyParts3D (muscle and bone
forms) are checked and in use; FLAME is out (owner, 2026-11-20).

## 1. Photorealism

1. **Scanned face into the default character.** Done 2026-11-21 (`target/agents/face-scan/STATUS.md`,
   stills `target/review/face/r2-final-*.png`): the mean face of 8 HSRD-100 people is on in the default
   template, with a rounded skull matching ANSUR head length, breadth and circumference, ears with a rim,
   and the nose-root seam faired (fold 88 -> 9 degrees). Round 3: the nape knob is a low, broad bump
   now; the head render test checks the shipped skin and passes. Round 4 (2026-11-22, stills
   `target/review/face/r4-*.png`): ears with rim, scapha, antihelix, concha, tragus and lobe, fold-free
   and ANSUR-sized (length 61.0 mm, breadth 34.0, protrusion 22.1; 25.6 degrees off the head; test
   `wp25_finished_ear_is_fold_free_and_ansur_sized`); a rounder forehead-to-temple curve; the pale
   column at the outer eye corner is gone. Still to do: the concha floor reads as a diamond-shaped pit in
   profile and a short crease runs forward from the ear's lower front; a faint tone change remains at the
   temple; the white sliver across the caruncle (a lid-margin highlight, not the tear line); natural
   asymmetry as the default (`identity.asymmetry` 0.35, tests that need a symmetric face set it to 0);
   a default procedural mole sits on the jugular notch since the torso UV repack (moles are placed in
   texture space: `skin_appearance/features.rs` should place them on the surface or keep them off
   landmarks).
2. **Face-shape controls from the scans.** Done 2026-11-21: `face.scan.shape_01..07` in the registry
   (`provisional`), each with a range measured to fold no triangle (shape_03 -1.6..0.6, the others about
   -1..1 or wider). Still to do: each component still mixes ancestry, sex, age and weight; ranges were
   measured one component at a time, not in combination.
3. **Body.** Done: measurements, pelvis, crotch, thighs and girths match ANSUR II. Still to do, in
   order:
   - **a female and a male body, both clearly adult** (owner, 2026-11-22): a Female / Male switch
     `identity.sex` (ADR-039: picks the genital variant, both shipped in the template, and the ANSUR
     women's or men's body and face), with `identity.sex_shape` as variation within a sex; the default
     character must read unmistakably as an adult of about 30 (today the face reads boyish).
     `identity.sex_shape=-1` barely changes the body (flat chest, hips
     hardly wider, no waist; lead's render 2026-11-22, `target/review/lead/female-*.png`) and the face
     stays boyish. The sex axis must give ANSUR women's and men's bust, waist, hip and shoulder girths
     and breadths (`data/anthropometry/ansur2.json`), breasts with volume, sex-typical fat distribution,
     and a female face direction; `body.torso.bust_size` and the other breast, waist and hip parameters
     must follow from it;
   - external genitalia, both variants (ADR-039): built 2026-11-21 (`generate/genitals.rs`; the body
     outside the crease loop is bit-identical with either variant; stills `target/review/body/`), but
     crude: the female site has a hard outline with a pointed top, thin ridges beside the penis root,
     from the front the penis hides most of the scrotum. `ac_wp11_1_template_head_ibl_and_shadows`
     now renders the shipped skin and passes (furnace error 0). UV packing: the face tile now packs better (nostril 1204 texels);
     the other tiles can't be repacked until `Mesh::face_uv_area` stops calling one-texel navel faces
     collapsed (f32 rounding);
   - surface anatomy: done 2026-11-22 for the torso (clavicles, jugular notch, sternocleidomastoid,
     pectoral border, sternum, costal margin, linea alba, iliac crest, ASIS, spine groove with erector
     columns, scapulae, PSIS dimples; smooth, tested for width and slope; stills
     `target/review/body/w10*`). Still to do: limb muscles; a faint crease at the front of each armpit;
     a small kink at the end of the shoulder outline;
   - hands and feet: done 2026-11-22 (knuckles, joint and palm creases, nails with folds and lunula
     regions, finger and toe pads, arch, malleoli, ANSUR hand and foot sizes). Still to do: toes are
     boxy; a back-facing sliver at a toe base; nails and creases are at the limit of the mesh density
     and need the skin side to shade the new nail regions;
   - pose correctives for shoulders, armpits, elbows, hips and knees. A `PoseCorrective` today fades
     in around one target rotation with a radius, which is weakest at the shoulder and armpit: blend
     several example poses per joint with radial basis functions and split twist from swing. Epic's
     OpenRigLogic (MIT, https://github.com/EpicGames/OpenRigLogic, `src/riglogic/rbf` and
     `src/riglogic/joints/cpu/twistswing`) is a tested reference to read, or to port with its MIT
     notice kept. The corrective shapes themselves are ours to make.
4. **Skin surface.** Done 2026-11-21 (`target/agents/skin-surface/STATUS.md`, stills
   `target/review/skin/`): `skin.*` values reach the compositor (CLI, editor, agent server), baked
   thickness and GPU-baked occlusion feed it, regional specular (Weyrich 2006) and sebum
   (Algiert-Zielińska 2017), specular occlusion for lights, low-frequency tone variation, procedural
   wrinkle maps for the 7 drivers, age spots, one bake per template, no corner glints. Still to do:
   region textures from the HSRD-100 scans (shadows and glints cleaned out); pores still invisible at
   portrait distance; age spots read as dots, not soft lentigines; nostril interior and lower-lip strip not re-checked on
   the final head; ADR for the regional specular / sebum sources; the dark-red ring in the
   eye socket under side light (item 1.9).
5. **Eyes.** Done: eye shading in CLI renders and the editor (GPU test), the eyelid guard in
   `Evaluator`, catchlights, iris depth, caruncle, tear line, meniscus, vessels, lid and lash shadow;
   cost +0.55 ms for an eye close-up at 1440p. Still to do: the eye looks computer-made next to a
   photo (mostly lashes and lid skin, items 1.4 and 1.7); a flat light patch at the outer corner in
   three-quarter view; parameter-driven eyeball size; rerun the shaders' own naga tests after the
   last shader changes.
6. **Mouth interior.** Done: lit cavity, wetter tongue, lower molars clear of the cheeks (0.84 mm) at
   every jaw opening. Still to do: the teeth read as flat tiles (enamel translucency, gums), the
   inside is still very dark, parameter-driven teeth.
7. **Hair, brows, lashes.** Done: strands on the head with placement from template regions, six
   hairstyle presets (short, medium straight, long wavy, curly, coily, afro), brows, lashes, peach
   fuzz and body hair, ribbon fallback, TAA stability, order independence; round 4 (2026-11-21,
   `target/agents/hair-render/STATUS.md`, stills `target/review/hair/r4-*.png`): lashes as smooth arcs
   (150 upper in 4 irregular rows, 75 lower, uneven clumps, none crumpled at the corners, none in the
   eye or skin), a soft hairline with baby hairs, parts on medium straight and long wavy, looser waves,
   an afro without a peak, guides blended in world directions (no strays), hair hidden behind the head
   culled (close-up 12.7 -> 9.9 ms at 1440p, GPU idle). Still to do:
   - close-up hair still about 10 ms at 1440p (frame 11.5 ms): cull hair hidden behind hair, or bin
     fragments per screen tile;
   - lashes: about 1 mm of skin shows along the lash line in places; lighter than in a photo in
     three-quarter view; checked against anatomy figures, not a photograph;
   - the forehead reads high under the hairline; medium straight shows a stepped two-layer cut from
     behind;
   - peach fuzz does not show even back-lit at close range, so it stays off by default.
8. **Expressions.** Round 2 done 2026-11-21 (`target/agents/expression2/STATUS.md`, stills
   `target/review/expression/`): all 89 controls pass the range-of-motion sweep, none of the 34 presets
   turns a triangle over, the smile no longer shows the mouth lining at the corners, the open mouth no
   longer stretches a skin sheet across the opening, the neutral upper lip has no folds, and
   `ExpressionDrive` drives the face from the document's sliders (clamped to each parameter's range,
   tested). 2026-11-22: contempt presets (one-sided AU12 + AU14, low/medium/high), 7 viseme
   parameters, `humanart-cli render --preset NAME` for every emotion and viseme. Still to do: sadness,
   disgust and contempt read weakly (contempt is a slight one-sided corner pull even at high: stronger
   AU14 shape needed), wrinkles do not appear with the expressions (no authored maps; disgust shows no
   nose wrinkle); `batch` cannot apply built-in presets; checked against FACS from memory, not against
   reference photographs.
9. **Light and camera.** Done 2026-11-21 (`target/agents/light-camera/STATUS.md`, stills
   `target/review/light/s4-*.png`): shadow maps fitted to the view (about 0.25 mm per texel on the
   face), soft penumbrae that grow with distance, ambient occlusion and contact shadows, 13 rigs
   (three-point, Rembrandt, loop, clamshell, rim, sunset and the older ones), Sunset and StudioRoom
   environments, 5 portrait camera presets with thin-lens depth of field, studio paper backdrop by
   default in the CLI. Viewport 2.8 ms at 1440p (budget 8 ms; GPU idle), occlusion at half resolution
   (0.33 ms), depth of field off in the viewport unless a lens is set. Still to do: depth of field costs
   3.6 ms in Review and Still; the overcast rig is about a stop dark; the
   occlusion is subtle (the depth buffer misses fold undersides); the dark-red ring in the eye socket
   under side light is the subsurface blur's (it weighs taps by screen offset only: proposal in the
   light-camera STATUS, for the skin); **`--test stills`
   `ac_wp11_3_tiled_8k_still_equals_the_untiled_render` fails since this work**: 1926 samples differ by
   more than one code value (bound 1327 of 132 M); likely cause: the shadow map is fitted to each tile's
   own view (`RenderConfig::shadow_focus`), so tiles get different shadow texel grids; a still should fit
   it once to the whole frame.
10. **How realism is judged.** The twelve review scenes (portrait front and three-quarter, macro skin,
   macro eye, macro lips and mouth, harsh side light, back light, hair back and profile, expression
   sweep, skin-tone range, age range, full body in poses, crowd), rendered from the CLI and the editor,
   scored by the owner on a written rubric (skin colour, translucency, specular, micro-detail,
   repetition, eye life, mouth, hair, expression, proportions, seams), next to licensed reference
   photographs. Automated metrics (colour difference, SSIM, seams, tiling, penetration) run with them.
   The owner's reference image `target/reference/char-01.png` stays the target for the body controls.

## 2. Editor

1. **Open the editor window once** (`cargo run -p humanart`): it has never run on screen (B-009).
2. Application shell: all modes usable by mouse and keyboard, property grid from the registry with
   search, history, provenance and proposal views, compare view and colour scopes, profiler overlay,
   preferences, crash recovery on relaunch.
3. Sliders: done 2026-11-22 (`target/agents/sliders/STATUS.md`): a test sets every registry parameter
   and checks the character changes (`crates/doc/src/registry/tests/effect.rs`); command results carry a
   plausibility score and an `implausible` warning; viseme parameters. Still to do: **97 of 293
   parameters do nothing** (eye, mouth and hair materials and grooms never fed from the document,
   45 `skin.features.*` read only from a skin layer, 5 without a skin target, gaze): being wired by the
   agent engine and app (list `KNOWN_DEAD` in that test).
4. Sculpt, paint and groom tools (WP-20): GPU brushes, symmetry, falloff, pen pressure, masks, locks.
5. Comparison, calibration and photo-guided fit (WP-22): colour-checker calibration, landmark
   annotation, camera and parameter fit from a photo.
6. Agent server and CLI (WP-23): done 2026-11-21, including the CLI commands `serve` (stdio, loopback
   HTTP with a token), `batch`, `turntable`, `validate` and the editor hook (`crates/app/src/
   agent_host.rs`: grant starts the server and admits the agent, pause/revoke drive its gate). User
   docs for the commands: text in `target/agents/cli-hook/STATUS.md`, to go into the user
   documentation (WP-36). Still to do: quality report and metrics (WP-24).

## 3. Data pipeline

Done: scan import, registration onto the template and the face model from 8 people
(`humanart-tools` `capture`, WP-25 and the face part of WP-27). Still to do: appearance capture and
fitting from the HSRD-100 photos (WP-26), a body model from data (WP-27 rest), consent, withdrawal
and provenance records. The scan-derived face model is published under HSRD-100's CC BY 4.0 with
the credits in `DATA-ATTRIBUTION.md` (ADR-056).
HumanArt itself is MIT OR Apache-2.0 (ADR-055).

## 4. Shipping

Export package and glTF (WP-29), runtime library with editor parity and crowds (WP-30), capability
fallbacks and low-memory behaviour (WP-33), large fuzz runs and server hardening (WP-34), user and
format documentation (WP-36), release packaging (WP-37), sample project (WP-38), 8-hour soak test
(WP-39), macOS and Linux builds (B-003). LOD package sections carry landmarks (ADR-021).

DNA export: write a character as a MetaHuman `.dna` file so tools that read DNA (MetaHuman Animator
and the Maya and Houdini plugins) can drive it. Needs our expression controls mapped onto theirs.
The format comes from OpenRigLogic's DNA library (MIT, https://github.com/EpicGames/OpenRigLogic;
port with its notice kept, no C++ dependency); an importer, if wanted, is a parser under `[ROB-01]`.
Only the format: MetaHuman assets (the OpenRigLogic sample content on Fab, MetaHuman Creator
exports) stay out of `data/` and the template (the Fab licence forbids redistributing an asset on its
own, the sample listing forbids use with AI, and there is no consent or provenance record for them).
"MetaHuman" may describe compatibility, never name a product or feature (OpenRigLogic
`TRADEMARKS.md`).

## 5. Performance

Benchmark suite and per-target report (WP-32): `humanart-cli bench --scenario 10` (the UI scene) exists
since 2026-11-21, scenarios 1-9 do not. Known gaps: close-up hair 9.9 ms at 1440p (item 1.7); skin
detail pass fits its 2 ms budget only up to 38 % face coverage; each journal append on Windows makes
34 small allocations (section 7). Done 2026-11-21 (`target/agents/loose-ends/STATUS.md`): GPU picking
reports the triangle, one `block_on`, GPU waits have a time limit, all test binaries share one GPU
device (ADR-040), the document side of the autosave no longer allocates.

## 6. Loose ends

- `ac_wp23_3_headless_render_matches_the_editor_viewport` (agent crate) failed once in 10 runs (a few
  pixels above 8/255; the other 9 runs differ by 0). Cause unknown. Ruled out: the hair coverage
  check-then-add in `shaders/hair/raster.wgsl` (it saturates at 16, a difference below 1e-6). To try:
  fragments with equal depth writing `hair_records` in a different order; the shared GPU device.
- `skin::tests::ac_wp13_3_partial_rect_cost_scales_with_area` (render lib, GPU timing ratio, bound
  2.0x) fails when other GPU tests run beside it (2.05x and 2.39x with 2 test threads) and passes alone:
  run GPU timing tests serially or mark them.
- `tools/fontbake/bake.sh` can call `humanart-cli fontbake --dir` now; then delete
  `crates/ui/examples/fontbake.rs`.
- Identity axes use a pooled standard deviation; ADR-046 asks for per-sex values.
- Shoulder width measured as 351 vs 382 mm by two agents: recheck on the current template.
- UV slivers that lock LOD vertices; low UV packing coverage.
- Rename the type `humanart_model::eval::rig::RigLogic` (15 uses, ADR-031), e.g. to `ControlRig`:
  Epic lists RigLogic as its trademark (OpenRigLogic `TRADEMARKS.md`). It is a Rust type, not a
  registry id, so renaming it changes no stored name.
- The embedded face model `data/template/face/face-model.hafm` cannot be rebuilt byte for byte from the
  8 current fits: a rebuild shifts the mean face by about 1 mm and turns components 02 and 04-07, which
  would change those sliders. The embedded file stays until the builder reproduces it (measured in
  `target/agents/face-scan/hafm-diff.txt`).

## 7. Waiting for the owner

- File extensions `.anthropos`, `.anchar`, `.antmpl`: keep them or rename to HumanArt ones.
- Skin reference: six ITA skin-tone groups in place of Fitzpatrick I-VI, and the colour tolerances the
  agents set after seeing their results.
- Skin defaults: `skin.haemoglobin` 0.3 -> 0.5 and `skin.eumelanin-ratio` 0.7 -> 0.75 to match measured
  medium skin.
- Autosave journal: the document side no longer allocates while a slider is dragged (74 of 300 frames
  before, 0 now). What is left is Windows itself: each append to the journal file makes 34 small
  allocations (path checks, opening the file). Zero needs either an open file handle kept by the
  platform layer (can go stale if the file is renamed or deleted from outside) or an open-handle
  append in the core file-system interface. Which one, or is 34 small allocations every 100 ms fine?
- Skin budget: 2.0 ms for a face in view, or for a face filling the frame?
