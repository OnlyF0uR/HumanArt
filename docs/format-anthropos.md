# The HumanArt character project (`.anthropos`)

A character is a project directory `[DOC-20]`: a line-oriented text document, a store of
content-addressed binary blobs, a command journal, a render cache and reference files. This document
specifies every file byte by byte, plus the parameter registry files under `data/template/params/`
that give the document's parameters their meaning. Implementation: `crates/doc/` (`humanart_doc::text`,
`document`, `migrate`, `blob`, `journal`, `project`, `registry`).

```text
<root>/character.anthropos   the document (text, section 1-3)
<root>/blobs/<hash>          one file per blob (section 4)
<root>/journal/journal.log   commands since the last full save (section 5)
<root>/renders/              cached renders; owned by the renderer, never read by the document
<root>/refs/                 reference photos and annotations; Reference paths are relative to it
```

All text is UTF-8 without a byte order mark, lines end in LF. All binary integers are
little-endian. Hashes are MurmurHash3 x64_128 with seed 0 (`humanart_core::hash::hash128`), written
as 32 lower-case hexadecimal digits (`h1` then `h2`, most significant digit first). Ids
(`humanart_core::id::Id128`) are 128-bit random numbers written the same way; the all-zero id is never
used. Writers never emit a hash of 32 zeros: [the empty content's hash](#4-blobs) means "no blob"
and is written `null`.

## 1. Document text grammar `[DOC-21]`

```abnf
file        = header *( blank / comment / section )
header      = "humanart-character" SP version EOL       ; first line, no blank or comment before it
version     = "0" / ( %x31-39 *9DIGIT )                  ; decimal u32, no leading zero
section     = "[" kind [ SP key ] "]" EOL *( blank / comment / entry )
entry       = key *SP "=" *SP value *SP EOL
kind        = %x61-7A *63( %x61-7A / DIGIT / "_" )        ; [a-z][a-z0-9_]*, <= 64 bytes
key         = ( ALPHA / DIGIT ) *127( ALPHA / DIGIT / "." / "_" / "-" )   ; <= 128 bytes
value       = <one JSON value, RFC 8259, on one line>    ; strict per ADR-010
blank       = EOL
comment     = "#" *( %x20-10FFFF / HTAB ) EOL
EOL         = LF / CR LF
```

- A line is at most `TextLimits::max_line_bytes` (16 MiB) long; a file has at most 2^20 sections
  and a section at most 2^20 entries; nesting inside a value is at most 16 levels; the whole text at
  most 256 MiB. Every limit is checked before the item is stored `[ROB-01]`.
- Entry lines may not be indented (a leading space makes the key invalid). The value is everything
  after `=` with the surrounding spaces removed; JSON's compact form never contains a line end
  because strings escape it.
- No two sections share both kind and key; no section has two entries with the same key.
- Entries may come in any order; they are kept sorted by key (byte order).

**Canonical form.** The writer emits the header, then for every section one empty line, the
section line and its entries in ascending byte order of the key, each as `key = value` (one space on
each side of `=`) with the value in the compact JSON of `humanart_io::json` (object members in their
stored order, integers in full, floats as the shortest digits that read back exactly and always with
a `.` or an exponent, the minimal string escapes), and nothing else: no comments, no trailing
spaces, no CR, a final LF. Parsing a canonical file and writing it gives the same bytes
`[AC-WP07-1]`; any other accepted file is rewritten in canonical form.

## 2. Document sections, version 1

The header is `humanart-character 1`. Sections appear in this order (the reader accepts any order
and rewrites it canonically); list sections keep their relative order, which is meaningful for the
layer stacks (bottom first). Every listed key is required; unknown sections and keys are errors.
Names are 1 to 256 bytes of any text; *template names* (regions, joints, landmarks, generator
names) follow the template's name rule (ASCII letters, digits, `.`, `-`, `_`, starting with a
letter or digit, at most 64 bytes). *Author* values name an `[author <id>]` section.

| Section | Key | Entries (value type) |
|---|---|---|
| `[meta]` | - | `created` (integer, Unix ms), `id` (id string), `modified` (integer, >= created), `name` (string), `template` (template name), `template_version` (integer, u32) |
| `[settings]` | - | `protect_human_edits` (bool, `[LOCK-02]`; default true when the section is absent) |
| `[author <id>]` | author id (key syntax), sorted | `kind` (`"human"` / `"agent"`), `name` (string). At least one author. |
| `[params]` | - | `<parameter id> = <value>` for every parameter that differs from its registry default `[DOC-02]`; value forms in section 3.3. Read-only parameters are never stored. Default values in a hand-written file are dropped. |
| `[param_authors]` | - | `<parameter id> = <author>`: who last changed each stored value. Entries without a stored value are dropped. |
| `[pose]` | - | `<joint name> = [x, y, z, w]`: the current preview pose `[DOC-09]`, unit quaternions (norm within 1e-6) relative to rest |
| `[sculpt_layer <id>]` | layer id | `author`, `blend` (`"additive"`), `created_by_command` (id string or null), `locked` (bool), `name`, `opacity` (number 0..1), `payload` (hash or null), `region` (region name or null for the whole body), `visible` (bool) `[DOC-03]` |
| `[skin_layer <id>]` | layer id | `author`, `blend` (`"mix"`, `"multiply"`, `"add"`, `"overlay"`), `channels` (array of channel names, sorted by the order below, no repeats), `locked`, `name`, `opacity` (0..1), `region` (or null), `source` (object, below), `visible` `[DOC-04]` |
| `[groom <id>]` | groom id | `author`, `guides` (hash or null), `kind` (`"scalp"`, `"eyebrows"`, `"eyelashes"`, `"facial"`, `"body"`), `locked`, `maps` (object: map name to hash or null; names `clump`, `color`, `curl`, `density`, `direction`, `frizz`, `length`), `name`, `params` (object: key-syntax name to number), `seed` (integer), `visible` `[DOC-08]` |
| `[expression_preset <id>]` | preset id | `name`, `weights` (object: expression parameter id to number within its range) `[DOC-09]` |
| `[pose_preset <id>]` | preset id | `joints` (object: joint name to `[x, y, z, w]` unit quaternion), `name` `[DOC-09]` |
| `[lock <id>]` | lock id | `kind` (`"region"` / `"parameter"`), `owner` (author), `target` (region name / registered parameter id) `[DOC-10]` |
| `[pin <id>]` | pin id | `landmark` (landmark name), `owner` (author), `position` (`[x, y, z]` metres) `[DOC-10]` |
| `[snapshot <id>]` | snapshot id | `author`, `created` (integer, Unix ms), `name`, `state` (hash, not null) `[DOC-11]` |
| `[attachment <id>]` | attachment id | `author`, `binding` (`"rigid"` / `"skinned"`), `format` (`"obj"` / `"glb"`), `joint` (joint name), `materials` (hash or null), `mesh` (hash, not null), `name`, `rotation` (`[x, y, z, w]` unit quaternion), `scale` (`[x, y, z]`), `translation` (`[x, y, z]` metres), `visible` `[DOC-12]` |
| `[reference <id>]` | reference id | `hash` (content hash of the file at import, or null), `kind` (`"photo"`, `"landmarks"`, `"color_target"`, `"lighting"`), `name`, `path` (relative to `refs/`: `/`-separated segments of letters, digits, `.`, `_`, `-`, space; no `.` or `..` segment; at most 512 bytes) `[DOC-13]` |

The writer always emits `[meta]`, `[settings]`, `[params]`, `[param_authors]` and `[pose]`, even when
empty, so that changes to them are entry edits. Ids are unique within each list.

Skin channel names, in their sort order: `albedo_tint`, `albedo_multiply`, `specular`,
`roughness`, `normal_detail`, `height`, `translucency`, `sss_color`, `sss_radius`, `freckle_mask`,
`mole_mask`, `blemish_mask`, `redness`, `cavity`.

Skin layer `source` objects (members in this order, no others):

```text
{"generator": <template name>, "kind": "procedural", "params": {<name>: <number>, ...}, "seed": <integer>}
{"kind": "painted", "texture": <hash or null>}
{"kind": "captured", "set": <hash or null>}
```

Procedural skin, eye and mouth settings `[DOC-05]`..`[DOC-07]` and the current expression weights
`[DOC-09]` are registry parameters and live in `[params]`. The command journal `[DOC-14]` is a
separate file (section 5).

A **snapshot state** blob holds the canonical text of the document at that moment without its
`[snapshot]` sections, so snapshots never nest.

## 3. Edits, versions and migrations

### 3.1 Edits

Every change of a document is a list of edits on its text form (`humanart_doc::text::Edit`): set,
add or remove one entry of a section (with the value expected before and the value after), insert a
whole section at an index, or remove one (with its full content). Applying checks each edit's
expectation and is all-or-nothing; the inverse list (each edit inverted, in reverse order) restores
the previous text exactly. `Document::apply_edits` additionally requires the result to be a valid
document in canonical form. The command system (WP-08) stores these as its forward and inverse
payloads.

### 3.2 Versions `[DOC-23]`

The header's version is the format version. A reader of version *n* migrates files of versions
1..*n*-1 through a chain of text-to-text functions (`humanart_doc::migrate::MIGRATIONS`, step *i*
turns version *i*+1 into *i*+2) and refuses newer versions. Version 1 is the first version, so the
chain is empty. Each version keeps a fixture `crates/doc/tests/fixtures/document-v<n>.anthropos`
that the tests load through the chain and the project loader.

### 3.3 Parameter values in `[params]`

| Registry kind | JSON value |
|---|---|
| `scalar` | number |
| `vector` | array of `dimension` numbers |
| `color` | array of 3 numbers (linear Rec.709) |
| `enum` | string, one of the entry's `choices` |
| `bool` | `true` / `false` |
| `curve` | array of 2..64 `[x, y]` pairs, `x` strictly increasing in [0, 1] |
| `asset_ref` | hash string, or `null` for none |

Numbers must be finite and within the entry's `min`..`max` (each component, each curve `y`).

## 4. Blobs `[IO-30]`, `[DOC-22]`

A blob file is named by the hash of its *content* (not of the file) and holds:

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | the bytes `48 41 42 4C` (`HABL`) |
| 4 | 1 | version | `1`; any other value is refused |
| 5 | 1 | flags | bit 0: the payload is an LZ frame; other bits `0` |
| 6 | 2 | reserved | `0` |
| 8 | 8 | content length | u64; at most `BlobLimits::max_bytes` (2 GiB) |
| 16 | - | payload | the content (flag clear: exactly *content length* bytes) or one `HALZ` frame (`docs/format-lz.md`) that decompresses to exactly *content length* bytes, with nothing after it |

Reading checks the header, the length limit before allocating, the LZ frame (its own checksum and
limits), the decoded length and finally that the content hashes to the file name; any mismatch is
an error naming the blob. Writers compress with `Compression::Auto` by default: content of at least
256 bytes is stored as LZ when that saves at least an eighth.

The empty content hashes to all zeros, means "no blob", and is never written. Blobs are immutable:
storing content that already exists writes nothing. Each blob is written atomically (temporary file
`<name>.tmp~`, then rename) and before the document that references it `[SAVE-02]`. Garbage
collection deletes every blob file not reachable from the document, the blobs referenced by its
snapshot states, open proposals and other explicit roots, plus leftover `*.tmp~` files; files whose
names are not 32 hexadecimal digits are left alone.

## 5. Journal `[DOC-14]`, `[SAVE-01]`

```abnf
journal  = header *record
header   = "humanart-journal" SP "1" SP document-id SP base-hash LF
record   = seq SP crc SP payload LF
seq      = "0" / ( %x31-39 *19DIGIT )        ; u64, consecutive records differ by exactly 1
crc      = 8( DIGIT / %x61-66 )              ; CRC-32 (IEEE) of the bytes `seq SP payload`
payload  = <one compact JSON value>          ; defined by the command system (WP-08)
```

`base-hash` is the hash of the bytes of `character.anthropos` the records apply to. A save writes the
document, then replaces the journal (atomically) with a header naming the new text. The editor
appends one record per committed command. On load: no journal means nothing to recover; a header
that names another document or another text hash is stale and not replayed; records in a journal
that continues the loaded text mean the last session did not end with a save, and they are replayed
onto it. A damaged last record (missing LF, bad CRC, bad JSON) is a write torn by a crash: it is
dropped and cut off before new records are appended. A damaged record followed by intact ones, or
a gap in the sequence numbers, is corruption and an error. Limits: 1 GiB per journal, 64 MiB per
record, 2^24 records, 64 levels of JSON nesting.

### 5.1 Journal records `[SAVE-01]`

Each record's payload is one compact JSON object `{"op": <op>, ...}` (`humanart_doc::save::Record`),
appended before the change is reported as done. Every record carries the exact edits it made, so
replay depends on nothing but the saved text (ADR-029). The ops:

| `op` | Members | Meaning |
|---|---|---|
| `commit` | `entry` (a history entry, below) | A command was applied; a gesture's first command is committed with its entry open |
| `extend` | `id`, `edits`, `last`, `elements` | The edits a gesture added since the last record, folded into its open entry; written at most every journal interval (default 100 ms) and when the gesture ends (the batching of slider drags) |
| `seal` | `id` | The gesture ended |
| `undo` | `id`, `edits` (the inverse that was applied) | Plain undo of the top entry |
| `redo` | `entry` | The entry applied again |
| `accept` | `proposal`, `entries` | Proposal commands merged into the main history `[PROP-02]` |

A **history entry** (`Entry::to_json`, also used in proposal files) is an object with exactly these
members (others are rejected):

```text
{"id": "<32 hex>", "author": "<author id>", "kind": "<command kind>", "label": "<label>",
 "time": <unix ms>, "last": <unix ms of the last extension>, "edits": [<edit>...],
 "reverts": "<32 hex>" | null, "open": "<coalescing key>" | null, "elements": <n>}
```

`reverts` names the entry a selective undo reverts (`kind` `"undo"`, `[CMD-04]`); `open` is the
coalescing key while a gesture keeps the entry open `[CMD-03]`; `elements` counts the elements strokes
touched. An **edit** (`humanart_doc::command::edit_to_json`) is one of:

```text
{"op": "set", "section": <kind>, "section_key": <key>?, "key": <entry key>,
 "before": <value>?, "after": <value>?}            ; absent before/after: entry created/removed
{"op": "insert" | "remove", "index": <n>, "section": <kind>, "section_key": <key>?,
 "entries": {<entry key>: <value>, ...}}            ; a whole section inserted/removed at index
```

Edits are read under `CommandLimits` (edit count, section index and values per section). Replay
(`humanart_doc::save::replay`) applies the records in order, all or nothing, and rebuilds the history
as far as the journal reaches (an undo or redo of a command from before the last save changes the
document but has no history entry to move over). When the records of a journal that continues the
saved text cannot be replayed, the session moves the journal aside to `<journal path>.damaged`, opens
the saved text and reports `JournalOutcome::Unreplayable`.

### 5.2 Proposals `[PROP-04]`

An open proposal is stored as `proposals/<id>.json` in the project, written atomically after every
change, so it survives a restart or a crash:

```text
{"format": "humanart-proposal", "version": 1, "id": "<id>", "author": "<author id>",
 "name": "<name>", "created": <unix ms>, "base": "<document text>", "entries": [<entry>...]}
```

`base` is the canonical text of the main document when the proposal was opened; `entries` are the
proposal's commands as history entries (5.1). Reading checks the file size and the JSON against
`ProposalLimits` (defaults: 1 GiB per file, 64 levels of nesting, 65 536 commands, the default text
and command limits) before anything is built, and re-applies every entry to the base to rebuild the
branch document. A file that fails is reported as damaged (`OpenReport::damaged_proposals`); the
other proposals still load. Accepting or discarding a proposal removes its file.

### 5.3 Autosaves `[SAVE-03]`

Autosaves are additional full copies of the document text in `autosave/`, named
`autosave-<n>.anthropos` with `n` a 20-digit, zero-padded, increasing number (the newest is the
highest). Each is checked like a full save (its blobs must already be in the project's store) and
written atomically; then all but the newest `keep` (at least 1) are deleted. Files in `autosave/`
whose names do not match the pattern are left alone. The session writes one after N changes and
after an idle time (defaults: every 50 changes or after 30 s without changes, keeping 5). Autosaves
are recovery points; they never replace the user's own save (`[SAVE-02]`: full saves write the blobs
first, then the text atomically, then a fresh journal).

## 6. Parameter registry files `[REG-01]`, `[REG-02]`

Every `*.json` file in `data/template/params/` (read in name order) is one JSON object:

| Member | Rule |
|---|---|
| `format` | `"humanart-params"` |
| `version` | `1` |
| `description` | optional string |
| `defaults` | optional object: entry members applied to every entry that does not set them (not `id`, not `sides`) |
| `parameters` | array of entries |

Entry members (after `defaults` are merged in; `null` counts as absent):

| Member | Rule |
|---|---|
| `id` | required; dot-separated segments of `a-z`, `0-9`, `_`, at most 96 bytes; unique over all files; never renamed or reused |
| `display_name`, `description` | required; the description says what the parameter does visibly (at least 16 characters) |
| `group` | required; `/`-separated words, the first one of `identity`, `face`, `body`, `expression`, `skin`, `eyes`, `mouth`, `hair`, `brows_lashes`, `facial_hair`, `lighting_camera` `[REG-03]` |
| `region` | optional template region name |
| `kind` | `scalar`, `vector`, `color`, `enum`, `bool`, `curve`, `asset_ref` |
| `dimension` | vectors: 2..16 (required); colours: 3; others: 1 |
| `choices` | enums: non-empty list of unique words; others: none |
| `unit` | a word (`coefficient`, `weight`, `fraction`, `m`, `deg`, `linear_rgb`, `ratio`) or empty |
| `min`, `max`, `soft_min`, `soft_max`, `step` | numeric kinds only: `min` and `max` required, `min <= soft_min < soft_max <= max`, `0 < step <= max - min` (defaults: soft = hard, step = range / 100) |
| `default` | required; a value of the kind (section 3.3) within the range |
| `symmetry` | optional `{"kind": "none"}`, `{"kind": "mirror", "partner": <id>, "side": "left"/"right"}` or `{"kind": "sided", "side": ...}`; mirror partners name each other, sit on opposite sides and share kind, range, choices and default |
| `affects` | required non-empty list of `shape`, `expression`, `pose`, `skin`, `eyes`, `mouth`, `hair`, `view` |
| `evaluation` | required; `{"kind": k, "target": <template name>}` with `k` one of `component` (delta component weight), `expression` (expression target weight), `rig` (joint), `material` (material parameter slot), `groom` (hair generator parameter), `editor` (editor state); or `{"kind": "measurement", "landmarks": [a, b], "axis": "x"/"y"/"z"/null}` |
| `read_only` | true exactly for measurements `[REG-05]` |
| `locked_by_default`, `plausibility_weight` (>= 0), `since_version` (>= 1), `tags` (words), `provisional` (reason the range or default is not settled) | optional |
| `editor_only` | optional; if present must equal the computed value (group `lighting_camera` or evaluation `editor`) |
| `sides` | optional `true`: the entry becomes `<id>.l` and `<id>.r`, display names get ` (left)` / ` (right)`, `{side}` in `region`, `target` and landmark names becomes `L` / `R`, and the two are mirror partners |

`describe_parameters` (`Registry::describe_parameters`) returns `{"format":
"humanart-params-description", "version": 1, "count": n, "groups": [...], "parameters": [...]}` with
one object per parameter in id order, holding every member above except `sides`, with `null` where
a member does not apply, plus `editor_only` `[REG-04]`, `[AC-WP07-2]`. Each such object is itself a
valid registry entry that reads back to the same parameter.

`Registry::check_template` performs the `[REG-02]` cross-check against a template: every component,
expression and rig target, region and measurement landmark must exist in the template, and every
identity, semantic and asymmetry delta component and every expression of the template must have a
parameter. Material, groom and editor targets are counted as unchecked; the shader and generator
crates check those.
