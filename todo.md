# To do

What HumanArt still needs, most important first. This file replaces the original specification as the
plan. `PROGRESS.md` records what is done and the evidence for it, `docs/decisions.md` the decisions and
`docs/blockers.md` what is blocked. Requirement IDs such as `[EYE-05]` in code and tests are labels from
the original specification; `PROGRESS.md` keeps the evidence under them.

## Where we stand (2026-11-20)

The engineering base is broad and tested: codecs, GPU layer, shader system, document and commands,
CPU and GPU evaluation, render graph, UI toolkit, skin model calibrated against measured skin colour,
subsurface shading, bakers, LODs, texture compression. Since 2026-11-20 the eyes, mouth interior and
hair are drawn on the head in CLI renders and the editor, the body measurements match ANSUR II, and
a mean face of 8 scanned people exists. The character still reads as **computer-made**: the scanned
face is not in the default character yet, the skin is waxy, the lashes and teeth look artificial and
expressions read weakly. Data: HSRD-100 (face shape, skin detail) and BodyParts3D (muscle and bone
forms) are checked and in use; FLAME is out (owner, 2026-11-20).

## 1. Photorealism

1. **Scanned face into the default character.** The mean face of 8 HSRD-100 people
   (`data/template/face/face-model.hafm`, `generate/scan_face.rs`) is built but switched off: with it
   on, 13 model tests failed (generate anat_02, mod_03, mod_06, mod_07, ac_wp06_1, adr_039; identity
   ac_wp06, id_05, id_02 x2; expression sweep; eyelid sweep; hair_04 x2). Their messages were never
   read, because the run died on the memory runaway fixed on 2026-11-20. Switch on with
   `python target/agents/scan-data/py/apply3.py apply`, read the failures, fix them (likely:
   vertex-exact mirror symmetry in `scan_face::apply`, move `anatomy.eye_center` with the eyes,
   re-baseline the ANSUR head constants, raise `TEMPLATE_VERSION`); no limit is loosened. Then: the
   bump at the back of the neck where the face meets the body, the band across the cheek, the light
   line between the lips (teeth moved with the lips), ears (thin blades from the front), natural
   asymmetry, the forehead shading band.
2. **Face-shape controls from the scans.** 7 components from the 8 people move nose, lips, jaw and
   face width, but each mixes ancestry, sex, age and weight, and they fold triangles at full range.
   Bind them to the registry (`face.scan.shape_01..07`, proposal in
   `target/agents/scan-data/STATUS.md` step 6) with a safe range.
3. **Body.** Done: measurements, pelvis, crotch, thighs and girths match ANSUR II. Still to do, in
   order:
   - external genitalia, both variants (ADR-039): half built (`generate/genitals.rs`, design in
     `target/agents/template-body/STATUS.md`); the genital faces stay in the closed `body` part.
     Since this half-built state, two template tests fail that passed on 2026-11-19:
     `ac_wp11_1_template_head_ibl_and_shadows` (render; one sample with furnace error 1, likely a
     broken normal) and `ac_wp15_2_thickness_on_the_template` (tools; the nostril has 972 texels,
     at least 1000 required: the new genital UV charts probably took face texture space);
   - visible surface anatomy: clavicles, scapulae, pectorals, spine groove, limb muscles
     (BodyParts3D as reference);
   - hands (knuckles, nails, palm creases) and feet (arches, toes, nails, thin spots on the soles);
   - pose correctives for shoulders, armpits, elbows, hips and knees;
   - the square outline around the nipples, the nose seam.
4. **Skin surface.** Region textures for albedo, normal and displacement (from the HSRD-100 scans
   after their baked shadows and scanner glints are cleaned out; until then from measured statistics);
   pores and fine lines that read at portrait distance; roughness and oil maps so the skin stops
   looking waxy; cavity and specular occlusion; wrinkle maps for brow, forehead, crow's feet and
   nasolabial folds, driven by expressions; age (wrinkle onset, thinning, spots); the document's
   `skin.*` values must reach the compositor (today it always shows the defaults); convert the baked
   thickness (metres, f32) to the compositor's format. Skin-shading flaws seen in the eye and mouth
   close-ups: white glints at the inner eye corners, cream-coloured nostril interiors, a bright spot
   at the inner lip corner, a skin-coloured strip along the lower lip's inner line.
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
   fuzz and body hair, ribbon fallback, TAA stability, order independence. Still to do:
   - lashes: a regular upright fringe, a light fan in three-quarter view; darker lash line, less even
     spacing, checked against a photo;
   - medium straight looks like a wig with a hard hairline; long wavy reads as crimp, not waves; the
     afro has a spire on top; one stray strand at the back of the head (blend guides in world
     directions);
   - peach fuzz does not show at portrait distance, so it is off by default;
   - close-up hair costs 12.7 ms at 1440p (depth 6.5, records 3.3): bin fragments per screen tile or
     cull hidden strands, then measure.
8. **Expressions, round 2.** 6 of 89 controls fail the range-of-motion sweep: au06 cheek raiser right
   (1 triangle turned over), au16, au25, au27 (172 turned over), viseme stretch, viseme jaw-open-wide
   (109 turned over); the presets surprise, fear, anger and the visemes aa, E, ih, kk, SS, TH turn
   triangles over. In the smile, mouth-interior faces roll out at both corners (red dots). Make
   expressions read clearly, checked against FACS references; register and test `ExpressionDrive`
   (`crates/doc/src/expression_drive.rs`, written, not in `lib.rs`, no tests) and drive the face from
   the document's sliders; wrinkles appear with the expressions (the drivers pass; there are no
   authored wrinkle maps).
9. **Light and camera.** Studio lighting rigs and HDR environments (permitted licence or generated),
   contact shadows and ambient occlusion, camera presets for portraits, lens and depth of field. Drop
   the flat grey background for review renders.
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
3. Sliders: check that every registry parameter moves the character; wire plausibility scoring into
   command results; viseme parameters.
4. Sculpt, paint and groom tools (WP-20): GPU brushes, symmetry, falloff, pen pressure, masks, locks.
5. Comparison, calibration and photo-guided fit (WP-22): colour-checker calibration, landmark
   annotation, camera and parameter fit from a photo.
6. Agent server and CLI (WP-23): the server, its tools, security and integration tests are done; still
   to do are the CLI commands `serve`, `batch`, `turntable` and `validate` (design in
   `target/agents/agent-server/STATUS.md`; adds the in-house `humanart-platform` to the CLI) and the
   editor hook that starts the server and calls `Server::admit` when the human grants the link.
   Quality report and metrics (WP-24).

## 3. Data pipeline

Done: scan import, registration onto the template and the face model from 8 people
(`humanart-tools` `capture`, WP-25 and the face part of WP-27). Still to do: appearance capture and
fitting from the HSRD-100 photos (WP-26), a body model from data (WP-27 rest), consent, withdrawal
and provenance records; a data attribution file (needed before the scan-derived face ships).
HumanArt itself is MIT OR Apache-2.0 (ADR-055).

## 4. Shipping

Export package and glTF (WP-29), runtime library with editor parity and crowds (WP-30), capability
fallbacks and low-memory behaviour (WP-33), large fuzz runs and server hardening (WP-34), user and
format documentation (WP-36), release packaging (WP-37), sample project (WP-38), 8-hour soak test
(WP-39), macOS and Linux builds (B-003). LOD package sections carry landmarks (ADR-021).

## 5. Performance

Benchmark suite and per-target report (WP-32). Known gaps: close-up hair 12.7 ms at 1440p (item
1.7); skin detail pass fits its 2 ms budget only up to 38 % face coverage; the autosave journal
allocates every 100 ms; GPU picking returns no triangle; two `block_on` implementations and some
unbounded GPU waits; one GPU device per test binary instead of the shared one (ADR-040).

## 6. Loose ends

- A proposal opened before a human edit does not see that edit: an agent change to the same parameter
  gets no `needs_review` warning (document crate).
- `humanart-cli fontbake` and `bench` scenario 10.
- `Ply::to_mesh` reports mesh problems at offset 0.
- Identity axes use a pooled standard deviation; ADR-046 asks for per-sex values.
- Shoulder width measured as 351 vs 382 mm by two agents: recheck on the current template.
- `docs/architecture.md` names a `build-template` command and a template file that do not exist.
- UV slivers that lock LOD vertices; low UV packing coverage.
- `StreamingPool::update_external` (skips a zero-page upload).
- Clippy on `humanart-agent` and `humanart-cli` not run since the hair round.

## 7. Waiting for the owner

- HSRD-100 licence: the dataset page says CC BY 4.0, each scan's own metadata says "Commercial" with
  a price. Which one counts? (The mean face of 8 people avoids making one man's face the default.)
- File extensions `.anthropos`, `.anchar`, `.antmpl`: keep them or rename to HumanArt ones.
- Skin reference: six ITA skin-tone groups in place of Fitzpatrick I-VI, and the colour tolerances the
  agents set after seeing their results.
- Skin defaults: `skin.haemoglobin` 0.3 -> 0.5 and `skin.eumelanin-ratio` 0.7 -> 0.75 to match measured
  medium skin.
- Does the autosave's allocation every 100 ms break the no-allocation rule?
- Skin budget: 2.0 ms for a face in view, or for a face filling the frame?
