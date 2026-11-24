# The command line (`humanart-cli`)

`humanart-cli` does headlessly what the editor does on screen: it renders stills, exports the
character, runs agents and checks data. It drives the same engine as the editor's viewport, so a
still shows what the viewport shows (WP-36 user documentation; every option is listed by
`humanart-cli <command> --help`).

Build it once, then run it from the repository root (the default `--data` folder is `data`):

```sh
cargo build --release -p humanart-cli
target/release/humanart-cli --help
```

Every command takes `--json-errors` (errors as one JSON line on standard error). Exit codes: 0
success, 2 usage, 3 data, 4 GPU, 5 render, 6 file, 7 project, 8 job file, 9 failed steps or checks,
10 agent server.

## Characters without a project

Without `--project` a command works on a new character in memory: every parameter at its registry
default (the woman, `identity.sex` −1, with natural asymmetry 0.35). `--set ID=V,...` changes scalar
parameters by their registry id (`data/template/params/*.json`):

```sh
humanart-cli render --set identity.sex=1,identity.age=1.5 --out older-man.png
```

## Stills: `render`

```sh
# A head portrait under the neutral studio rig.
humanart-cli render --view head --out head.png

# A three-quarter portrait with an 85 mm lens and shallow depth of field, Rembrandt light.
humanart-cli render --camera head-and-shoulders --view three-quarter --aperture 2.8 --rig rembrandt --out portrait.png

# An expression preset, then a close-up of the mouth.
humanart-cli render --preset happiness.high --view head --out smile.png
humanart-cli render --preset viseme.oh --close-up mouth --width 960 --height 640 --out oh.png

# Macro stills: the left eye, the mouth, the nose.
humanart-cli render --close-up nose --hair none --out nose.png
```

- `--view` frames the body, the head, three-quarter or profile; `--camera` picks a portrait lens
  instead; `--close-up` frames the eye, the mouth or the nose from the `--view` direction.
- `--rig` chooses one of 13 light rigs; `--background` studio paper (default), the rig's own,
  flat grey or the environment.
- `--frames` is how many frames accumulate before the capture (32 by default): fewer frames leave
  the stochastic soft shadows grainy.
- `--hair none` removes all hair (also brows and lashes); `bald` keeps brows, lashes and fuzz.
- `--skin plain` draws one skin material instead of the surfaced skin, to tell a geometry problem
  from a texture one.

### Posed stills: `--pose`

`--pose JOINT=ROT,...` turns joints relative to their rest orientation, in each joint's own frame
(Y along the bone, Z toward flexion). `ROT` is axis letters with degrees, applied in order:

```sh
# Arms down to the sides, left elbow bent, right knee bent, left hand turned.
humanart-cli render --pose "L.upper-arm=z-43,R.upper-arm=z43,L.forearm=x100,R.shin=x90,L.hand=y70" --out pose.png
```

The arms of the template rest in an A-pose; lowering them is Z about the upper arm (negative on the
left, positive on the right). Elbows, knees and hips flex about X. Rolling the hand (pronation)
turns the forearm's twist joint with it, and the pose correctives keep the bent elbows, knees,
shoulders and hips in shape (ADR-057). Joint names are listed by the editor's pose panel and in the
skeleton of an exported glTF.

### A turntable: `turntable`

```sh
humanart-cli turntable --views 24 --camera head --out turntable/
```

## Export: `export`

```sh
humanart-cli export --set identity.sex=1 --out man.glb
humanart-cli export --expressions off --out shape-only.glb
```

Writes glTF 2.0 binary: the rest shape with the parameter values applied, the skeleton as nodes,
four skin weights per vertex, one primitive per mesh part and UV tile, every expression target as a
morph target (named in the mesh's `extras.targetNames`), skin colour from the pigment model. Not in
the file yet: skin textures, hair, and the combination and pose correctives (glTF cannot drive them;
they are named in the mesh's extras). Blender, Unreal and three.js import it; the morph targets drive
the face.

## Agents: `serve` and `batch`

`serve` runs the agent server (MCP) on stdio or on loopback HTTP with a token; an agent then edits
the character through the same commands as a person, inside proposals unless `--direct-edits`:

```sh
humanart-cli serve --transport stdio --project my-character --create
```

`batch` runs a job file of tool calls as an agent on stdio would and prints a JSON report:

```json
{"version": 1, "steps": [
  {"tool": "set_parameters", "arguments": {"values": {"identity.age": 1.5}, "reason": "older"}},
  {"tool": "apply_expression", "arguments": {"preset_name": "happiness.medium"}},
  {"tool": "describe_character"}
]}
```

```sh
humanart-cli batch --job job.json --project my-character --create --direct-edits --save
```

Without `--direct-edits` the agent's edits go into a proposal (`open_proposal` first, then the
edits with its `proposal_id`) that a person reviews in the editor.

`apply_expression` applies a preset stored in the document by name, or a built-in preset (the
names `--preset` takes) when the document has none of that name. `--keep-going` runs the remaining
steps after a failed one; the exit code is 9 when a step failed.

## Checks: `validate`

```sh
humanart-cli validate --project my-character --json
```

Checks the data folder (registry, template, presets) and a project without changing them.

## Benchmarks and fonts

`bench --scenario 10` measures the UI benchmark scene; the other scenarios are not written yet.
`fontbake` bakes the UI's distance-field fonts from `data/fonts` (`tools/fontbake/bake.sh` runs it).

## Without a graphics card

The rendering commands need a GPU adapter. On Linux without one, Mesa's software Vulkan driver
(lavapipe, `libvulkan_lvp.so`) works: a 540 × 675 head still takes about 90 seconds, most of it
building the template. Software rendering differs from a GPU in its last bits (signed zeros,
division rounding), never in what a still shows.
