#!/bin/sh
# check-deps.sh: fails when `cargo tree` shows a crate that docs/dependencies.md does not record
# [DEP-06]. The record is the fenced block under "## Recorded crates" in that file: one crate name
# per line. Regenerate it, after justifying the change in the document, with:
#
#   sh tools/check-deps.sh --list
#
# DEPS_DOC=<file> reads another record; DEPS_LIST=- reads the crate list from stdin instead of
# running cargo (both for the self-test).

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DOC="${DEPS_DOC:-$ROOT/docs/dependencies.md}"

# Every third-party crate of every target the workspace builds for (native and wasm32), by name.
list() {
    if [ "${DEPS_LIST:-}" = "-" ]; then
        cat
        return
    fi
    cd "$ROOT" || exit 2
    cargo tree --workspace --target all --prefix none -e normal,build 2>/dev/null |
        sed -e 's/ (.*//' -e 's/ v[0-9].*//' | grep -v '^humanart' | grep -v '^$' | sort -u
}

if [ "${1:-}" = "--list" ]; then
    list
    exit 0
fi

[ -f "$DOC" ] || { echo "FAIL deps: $DOC is missing"; exit 1; }
RECORDED=$(awk '/^## Recorded crates/ { on = 1; next } on && /^```/ { fence++; next } on && fence == 1 { print } on && fence >= 2 { exit }' "$DOC" | sort -u)
ACTUAL=$(list)
[ -n "$ACTUAL" ] || { echo "FAIL deps: cargo tree printed nothing"; exit 1; }

FAILED=0
for crate in $ACTUAL; do
    if ! printf '%s\n' "$RECORDED" | grep -qx "$crate"; then
        echo "FAIL deps: '$crate' is in cargo tree but not recorded in docs/dependencies.md"
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "check-deps: FAILED"
    exit 1
fi
echo "check-deps: ok ($(printf '%s\n' "$ACTUAL" | wc -l | tr -d ' ') crates, all recorded)"
