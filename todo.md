# To do

What HumanArt still needs, most important first. This file replaces the original specification as the
plan. `PROGRESS.md` records what is done and the evidence for it, `docs/decisions.md` the decisions and
`docs/blockers.md` what is blocked. Requirement IDs such as `[EYE-05]` in code and tests are labels from
the original specification; `PROGRESS.md` keeps the evidence under them.

## Where we stand (2026-11-23)

The engineering base is broad and tested: codecs, GPU layer, shader system, document and commands,
CPU and GPU evaluation, render graph, UI toolkit, skin model calibrated against measured skin colour,
subsurface shading, bakers, LODs, texture compression. The eyes, mouth interior and hair are drawn on
the head in CLI renders and the editor, the body measurements and both sexes' faces match ANSUR II,
and the default face is the mean of 8 scanned people. Since 2026-11-23: pose correctives and driven
twist joints, glTF export, the shadow acne that looked like a mole fixed, every crate's tests run
(on a software GPU: lavapipe renders work, about 90 s per head still, which is how tonight's visual
checks were made). The character still reads as **computer-made**: the skin is waxy, the lashes and
teeth look artificial, expressions read weakly, and the default woman with the default short hair
reads boyish (a longer default hair style for her is the owner's call). Data: HSRD-100 (face shape,
skin detail) and BodyParts3D (muscle and bone forms) are checked and in use; FLAME is out (owner,
2026-11-20).

## 1. Photorealism

**The quality bar (owner, 2026-11-24).** The owner's reference images (AI-generated stills and GTA 6
characters, kept out of the repository; not to be copied as likenesses) set the quality HumanArt must
reach easily: adult women of about 25 to 35, fit and toned with an hourglass build, high cheekbones
over slim cheeks, tapered jaws, full lips, narrow noses, large defined eyes with heavy lashes, arched
brows, warm skin with a sheen and real pores, hair with strands and flyaways (ponytails, bobs, long
layers), clothing and jewellery. Anything short of that is insufficient. Measured against it
(lavapipe renders, 2026-11-24) the character still reads as a smooth, bland mannequin. In order of
what closes the gap most:

1. **Face sliders too weak**: the semantic face axes move features by about 1 mm per unit (cheek
   fullness at 1.5 changes the jaw breadth by 0.9 mm); calibrate each to a population SD (Farkas's
   anthropometry) as the body axes are to ANSUR II, so the starter values can reach the references'
   features. Then a sculpted feminine-attractive face direction (brows, lids, lips, nose, cheek
   hollows) on top of the scanned mean, which averages 8 people into blandness.
2. **Skin**: pores at portrait distance, a sheen (oil, sweat) control that reads, makeup layers
   (liner, lash line, lipstick, contour), region textures from the HSRD-100 scans (item 4 below).
3. **Hair**: ponytail, bob, updo and long layered styles with flyaways and baby hairs; the six presets
   are smooth shells by comparison.
4. **Eyes and lashes**: iris detail, a wet lid line, lashes thick enough to read as lined eyes.
5. **Body surface**: abdominal definition, toned arm and shoulder relief, a navel and hip bones that
   read under light skin (limb muscles are listed under item 3).
6. **Clothing and jewellery**: every reference wears them; HumanArt has attachments (rigid or skinned
   meshes) but no garments. For the owner: in scope, and how (imported garments, a cloth tool)?

Done 2026-11-25 toward it (ADR-058, lavapipe renders): **makeup** (`skin.makeup.*`,
`skin_appearance::makeup`): winged liner, lower lash line, smoked shadow, lipstick with liner and
gloss, blush, contour and highlight, placed from the landmarks and generated into the skin
compositor over the face and eyes tiles; the woman starter wears it (starters take colours now).
**`face.shape.sculpt`**: one smooth field for a lean, tapered face (jaw angle 7 mm in and 3 mm up
per unit, cheek hollowed, chin forward, a crisp line under the jaw), fold-free in -1 .. 2; the
woman starter uses it at 1.3 instead of the cheek and jaw axes, whose large values left lumps over
the jaw and a crease beside the nose. Her lashes are shorter and sparser (they read as false-lash
brooms). Found on the way: the compositor draws every UV tile at 512² (`SKIN_TILE_SIZE`), so the
face tile (its islands span 506 mm per UV unit) is coloured at **1 mm per texel**: lip liner,
contour, freckles and moles are blurred by it (the lids, on the eyes tile, get 0.3 mm); a per-tile
size or a denser face tile is the fix. And the shader, the wrinkle maps and the feature layers took
each tile's millimetres per UV from its resolution over its target density, which the packer does
not reach (face 819 assumed, 506 drawn; scalp 1024 against 367): pores, fine lines, wrinkles and
feature sizes on the face were 1.6 times too fine. They take the measured scale now
(`Template::mm_per_uv`). Pores still do not read at portrait distance: the diffuse light is blurred
over millimetres by the subsurface scattering, so only the specular can show them, and the skin's
specular is weak in every rig. The lips' vertical stripes and scalloped border were the lip-fullness axes (a weight around 22
landmark points that dipped between them, pushed along the raw normals, which deepened the lips'
small ridges): now 1 inside the lip outline and smoothed (`target/review/makeup/mouth2.png`); the
nose-length axis folded the alar groove with the man's face (smoothed, longer ramp). Still wrong in
the renders: small eyes under puffy lids, the liner mostly hidden under the lash roots, lashes that turn
pale at portrait distance (dark in a close-up), a round, childlike face overall.

Compared against the reference images (2026-11-25, crops of their faces next to lavapipe renders;
`diag_starter_face_proportions` prints the starter's proportions against the neoclassical canons):
the references' faces are narrower (no axis narrowed the midface: **`face.shape.width`**, 3 mm a
side per unit outside the eyes; the woman at -2, bizygomatic 122.5 mm), with relaxed almond eyes
(ours stared: opening 12 mm, now 10.8 with `eyes.size` -1.6), low, bold, filled brows
(**`brows_lashes.brows.thickness`**, **brow pencil** makeup), dark lash lines (**mascara**,
`brows_lashes.lashes.thickness`), lighter irises (hazel), a mouth about 50 mm wide (the starter's
had grown to 54), and a jaw at about 0.5 of the face breadth (the sculpt at 1.4 had taken it to
0.43, gaunt; the sculpt now narrows the jaw half as much and hollows the cheek and lifts the
cheekbone more, so the three-quarter view shows a cheekbone over a hollow). A dewy finish
(oiliness 0.75, wetness 0.15) shows under a key light, little under the soft studio rigs. Still
behind them: the forehead is bare and high (their hair falls across it;
a swept fringe from the guides alone gave a thin veil, it needs a denser styled preset), the ears
show, the skin is matte (the highlight is stronger now), the eye framing is lighter than their
heavy liner, and the lower face is still soft. **Renderer bug** (2026-11-25): in three-quarter
views the scalp hair's edge at the temple shows square, stair-stepped blocks about 4 px wide
(`target/review/makeup/vL-34.png`); only with scalp hair (not bald), and the same with
`--occlusion off`, `--frames 1` and `--skin plain`, so it is in the hair raster or resolve, not
the AO, TAA or skin passes; the deep opacity and density-grid lookups are already filtered.

Done 2026-11-24 toward it: starter characters (`data/template/characters.json`, `humanart_doc::starter`):
a new character is the `woman` starter (the references' build and face direction, medium straight
hair; the CLI's `--character woman|man|average`, the editor's sample character, the agent host);
`average` keeps the ANSUR II mean woman. The midface breadth is fitted to ANSUR II per sex (133.8 /
142.6 mm, was 140 / 147: the face read round and young).

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
   temple; the white sliver across the caruncle (a lid-margin highlight, not the tear line).
   Done 2026-11-23: natural asymmetry is the default (`identity.asymmetry` 0.35); the "mole on the
   jugular notch" was shadow acne, not a mole (it stayed with plain skin and moved with the light): the
   shadow filter's receiver plane counted the rim of a concave hollow as a blocker of its floor; a
   curvature allowance (`SHADOW_CURVATURE` in `shaders/render/lighting.wgsl`) removes the specks in the
   notch and over the clavicles with the nose, lid and lip shadows unchanged (lavapipe renders, owner
   to confirm on a real GPU); moles keep clear of the jugular notch, nipples, navel and eye and mouth
   corners anyway (`feature_layers::SPOT_CLEARANCE`, test `todo_1_1_moles_keep_clear_of_landmarks`); the adult face of both sexes (2026-11-23, test
   `todo_1_3_both_sexes_have_adult_proportions`): menton-sellion now 112.6 / 122.1 mm (ANSUR II means
   113.1 / 122.6; the chin longer and 1 mm more prominent), nose protrusion 17.0 / 17.8 mm (was 15.7 for
   the woman), lid opening / fissure 0.42 (was 0.43; 0.41 read sleepy in a render), the skin under the chin
   lifted 3 mm (submental drop 3.3 / 2.0 mm). The sex switch fits the face height on the profile
   (sellion to menton found anew on each shape) instead of two fixed landmark vertices. Not yet looked
   at in a render.
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
     the other tiles can be repacked now: `Mesh::face_uv_area` works relative to the first corner and
     calls no face of the template collapsed (2026-11-23, `diag_uv_flipped_faces`: 0 faces);
   - surface anatomy: done 2026-11-22 for the torso (clavicles, jugular notch, sternocleidomastoid,
     pectoral border, sternum, costal margin, linea alba, iliac crest, ASIS, spine groove with erector
     columns, scapulae, PSIS dimples; smooth, tested for width and slope; stills
     `target/review/body/w10*`). Still to do: limb muscles; a faint crease at the front of each armpit;
     a small kink at the end of the shoulder outline;
   - hands and feet: done 2026-11-22 (knuckles, joint and palm creases, nails with folds and lunula
     regions, finger and toe pads, arch, malleoli, ANSUR hand and foot sizes). Still to do: toes are
     boxy; a back-facing sliver at a toe base; nails and creases are at the limit of the mesh density
     and need the skin side to shade the new nail regions;
   - pose correctives: done 2026-11-23 (ADR-057). `PoseDriver` solves the correctives of a joint by
     radial basis interpolation over swing, twist or the whole rotation (after OpenRigLogic, MIT
     notice kept; package section `PDRV`); the template derives 22 correctives (elbow 90 / 140, knee
     90 / 126, shoulder down / up / forward / back, hip flexed / extended / spread, both sides) from
     dual-quaternion skinning, Delta-Mushed with the volume put back at the elbow, knee and shoulder.
     Against plain skinning (test `todo_1_3_correctives_keep_volume_and_smooth_folds_at_bent_joints`):
     knee at 126 degrees 709 cm³ lost -> none, mean fold angle 4.05 -> 2.84 degrees; elbow at 126
     degrees 7.8 -> 2.4 degrees and its volume kept; arm raised forward 47 -> 8 faces turned over; hip
     flexed 57 -> 24. The twist joints are driven now (`eval::twist`): the forearm takes the hand's
     roll, the upper arm's twist joint undoes the arm's roll at the shoulder. Still to do: contact
     where the calf meets the thigh past 126 degrees and the elbow folds on itself at 140 (about 160
     faces turned in the crease); the hip gains volume (dual quaternions bulge); not yet looked at in
     a render (`humanart-cli render --pose L.forearm=x120` now poses joints).
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
   three-quarter view; parameter-driven eyeball size. (The shaders' naga tests pass after the last
   shader changes, 2026-11-23.)
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
   parameters, `humanart-cli render --preset NAME` for every emotion and viseme; 2026-11-23: the agent
   tool `apply_expression` (so `batch` too) applies a built-in preset by name when the document has no
   stored one of that name; AU14 pulls the corner 3.4 mm back (was 2.2) with a deeper dimple and the
   tissue bulging beside it; AU9 drives new nose wrinkle maps (`nose.L` / `nose.R`, crease across the
   nasal root and oblique lines on the sides; `humanart-cli render --close-up nose`). Still to do: in
   lavapipe renders sadness, disgust and contempt still read weakly at head framing (the nose lines
   show in the close-up only); the other wrinkle maps (0.2 mm deep) are faint too; checked against FACS
   from memory, not against reference photographs.
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
   more than one code value (bound 1327 of 132 M). Checked 2026-11-23 on lavapipe (1180 samples, mean
   0.0037 against 0.002): the shadow map is already fitted once to the whole frame (the renderer fits
   the focus with the full image's aspect), every noise source is seeded with the global pixel and the
   still's sample index, the occlusion pass is off in that test and its bust is not skin; the
   differences are scattered over the bust's soft shadows (a different grain in the penumbra), not
   at the seams. Cause still open. Separately, real skin needs a wider tile overlap than 32 pixels: the
   subsurface blur reaches 95 pixels on a face at 0.8 m in an 8K image (`still::subsurface_overlap`,
   `Scene::nearest_skin`; callers of `render_still` must size the overlap with them).
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
   docs for the commands: `docs/command-line.md` (2026-11-23; its batch example was run). Still to do: quality report and metrics (WP-24).

## 3. Data pipeline

Done: scan import, registration onto the template and the face model from 8 people
(`humanart-tools` `capture`, WP-25 and the face part of WP-27). Still to do: appearance capture and
fitting from the HSRD-100 photos (WP-26), a body model from data (WP-27 rest), consent, withdrawal
and provenance records. The scan-derived face model is published under HSRD-100's CC BY 4.0 with
the credits in `DATA-ATTRIBUTION.md` (ADR-056).
HumanArt itself is MIT OR Apache-2.0 (ADR-055).

## 4. Shipping

Export: glTF done 2026-11-23 (`humanart_model::export`, `humanart-cli export --set ... --out x.glb`): the
rest shape with the document's identity, the skeleton as nodes with inverse bind matrices, four
weights per vertex, one primitive per part and UV tile, the 94 expressions as sparse morph targets,
skin colour from the pigment model; test `exp_p1_the_character_exports_as_a_valid_skinned_glb_with_morph_targets`
(the in-house validator; the Khronos validator is not available here, B-007). Still to do: the baked
skin textures (GPU), hair as cards, combination and pose correctives (glTF cannot drive them; they are
named in the mesh's extras), the export package (WP-29 rest), runtime library with editor parity and crowds (WP-30), capability
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
- GPU tests that share the one device: `ac_wp13_3_partial_rect_cost_scales_with_area` takes the
  fastest of 15 rounds and the crate's timing lock (passed 2026-11-23 beside other tests). On a
  software adapter (lavapipe) three agent tests (`todo_1_4_resting_wrinkles_crease_the_face`,
  `todo_2_3_document_eye_and_mouth_materials_reach_the_feature_table`, the scripted MCP sessions)
  fail only when run beside the others (timeouts, a viewer that comes up without its wrinkles) and
  pass alone; the groom edit latency test cannot meet 16 ms there.
- Identity axes use a pooled standard deviation; ADR-046 asks for per-sex values.
- Shoulder width: settled 2026-11-23 by the sex switch's fitted gauge (biacromial 368.6 mm woman, 419.0 mm
  man against the ANSUR II means 365.3 / 415.7; test `todo_1_3_sex_switch_meets_the_ansur_means`).
- UV slivers that lock LOD vertices; low UV packing coverage.
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
