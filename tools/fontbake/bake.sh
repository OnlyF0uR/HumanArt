#!/bin/sh
# tools/fontbake/bake.sh: bakes the UI fonts of data/fonts/ into .anfont distance-field atlases
# [UI-03] with the library baker humanart_ui::fontbake (through the crate's `fontbake` example).
#
#   sh tools/fontbake/bake.sh            bake every shipped face into data/fonts/<face>.anfont
#   sh tools/fontbake/bake.sh <dir>      bake the faces found in <dir> into <dir>
#
# Faces: regular, semibold, italic (Inter), mono (JetBrains Mono): the Latin set of
# humanart_ui::fontbake::latin_charset at 32 texels per em, range 4; icons (Lucide): every mapped
# icon at 24 texels per em, range 3. Re-run after changing a source font or the baker, and record
# the result in docs/data-provenance.md.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR="${1:-$ROOT/data/fonts}"
cd "$ROOT"
cargo run --quiet --release -p humanart-ui --example fontbake -- "$DIR"
