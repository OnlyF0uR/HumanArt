#!/bin/sh
# check-tree.sh: keeps the tree honest (spec 15.7, 17.2 rule 6).
#
#   sh tools/check-tree.sh            static checks, then the cargo-backed checks
#   sh tools/check-tree.sh --static   static checks only (no cargo; what the self-test uses)
#
# ROOT=<dir> scans another tree (the self-test points it at trees with injected violations).
# Exit code 0 when clean, 1 when any check failed. Every failure is printed as "FAIL <check>: ...".

set -u
MODE="${1:-full}"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 2
FAILED=0

fail() {
    echo "FAIL $1: $2"
    FAILED=1
}

# All Rust sources of the workspace, and every hand-written text file a stub could hide in.
rust_files() { find crates -name '*.rs' -not -path '*/target/*' 2>/dev/null | sort; }
text_files() {
    {
        rust_files
        find shaders -type f -name '*.wgsl' 2>/dev/null
        find tools -type f \( -name '*.sh' -o -name '*.rs' -o -name '*.py' \) \
            -not -name 'check-tree.sh' -not -name 'check-tree-selftest.sh' 2>/dev/null
        find data -type f \( -name '*.json' -o -name '*.txt' -o -name '*.anparams' \) 2>/dev/null
    } | sort
}

# --- 1. No stubs [0.1.6] --------------------------------------------------------------------------
text_files | while IFS= read -r f; do
    grep -nE 'todo!|unimplemented!|\bTODO\b|\bFIXME\b|\bXXX\b' "$f" | while IFS= read -r hit; do
        echo "FAIL stubs: $f:$hit"
    done
done >"${TMPDIR:-/tmp}/humanart-check-stubs.$$"
if [ -s "${TMPDIR:-/tmp}/humanart-check-stubs.$$" ]; then
    cat "${TMPDIR:-/tmp}/humanart-check-stubs.$$"
    FAILED=1
fi
rm -f "${TMPDIR:-/tmp}/humanart-check-stubs.$$"

# --- 2. No unwrap()/expect() outside tests [ERR-03] -----------------------------------------------
# Test code is: everything from the first `#[cfg(test)]` of a file to its end (the tree's convention
# is one test module at the end of the file), and whole files under tests/, benches/ or named
# tests.rs. A call is allowed when one of the three lines before it carries an `INVARIANT:` comment,
# or when the file is listed (with a justification) in tools/unwrap-allowlist.txt.
ALLOW="tools/unwrap-allowlist.txt"
rust_files | while IFS= read -r f; do
    case "$f" in */tests/* | */benches/* | */tests.rs) continue ;; esac
    if [ -f "$ALLOW" ] && grep -qE "^$f([[:space:]]|$)" "$ALLOW"; then continue; fi
    awk -v file="$f" '
        /^[[:space:]]*#\[cfg\(test\)\]/ { exit }
        { line[NR] = $0 }
        /^[[:space:]]*\/\// { next }
        /\.unwrap\(\)|\.expect\(/ {
            ok = 0
            for (i = NR - 3; i < NR; i++) if (i > 0 && line[i] ~ /INVARIANT:/) ok = 1
            if ($0 ~ /INVARIANT:/) ok = 1
            if (!ok) printf "FAIL unwrap: %s:%d: %s\n", file, NR, $0
        }' "$f"
done >"${TMPDIR:-/tmp}/humanart-check-unwrap.$$"
if [ -s "${TMPDIR:-/tmp}/humanart-check-unwrap.$$" ]; then
    cat "${TMPDIR:-/tmp}/humanart-check-unwrap.$$"
    FAILED=1
fi
rm -f "${TMPDIR:-/tmp}/humanart-check-unwrap.$$"

# --- 3. Only allowed direct dependencies [DEP-01, DEP-02] -----------------------------------------
for toml in Cargo.toml crates/*/Cargo.toml; do
    [ -f "$toml" ] || continue
    # Every `name = ...` / `name.workspace = true` line inside a dependencies table.
    awk '
        /^\[/ { dep = ($0 ~ /dependencies(\]|\.)/ || $0 ~ /dependencies\]$/) ; next }
        dep && /^[A-Za-z0-9_-]+[[:space:]]*(=|\.)/ { n = $1; sub(/[.=].*/, "", n); gsub(/[[:space:]]/, "", n); print n }
    ' "$toml" | sort -u | while IFS= read -r dep; do
        case "$dep" in
            humanart-core | humanart-math | humanart-io | humanart-model | humanart-doc | humanart-gpu | humanart-shaders | humanart-render | humanart-ui | \
                humanart-tools | humanart-capture | humanart-agent | humanart-runtime | humanart-platform | wgpu | winit) ;;
            *) echo "FAIL deps: $toml depends on '$dep', which is not an allowed crate" ;;
        esac
    done
    if grep -qE '^build[[:space:]]*=' "$toml" || [ -f "$(dirname "$toml")/build.rs" ]; then
        echo "FAIL deps: $toml has a build script [PERF-R8]"
    fi
    if grep -qE '^proc-macro[[:space:]]*=[[:space:]]*true' "$toml"; then
        echo "FAIL deps: $toml is a proc-macro crate [PERF-R8]"
    fi
done >"${TMPDIR:-/tmp}/humanart-check-deps.$$"
if [ -s "${TMPDIR:-/tmp}/humanart-check-deps.$$" ]; then
    cat "${TMPDIR:-/tmp}/humanart-check-deps.$$"
    FAILED=1
fi
rm -f "${TMPDIR:-/tmp}/humanart-check-deps.$$"

# --- 4. Unsafe code only where the spec allows it [UNS-01, UNS-02] --------------------------------
for dir in crates/*/; do
    name=$(basename "$dir")
    root="$dir/src/lib.rs"
    [ -f "$root" ] || root="$dir/src/main.rs"
    [ -f "$root" ] || { fail unsafe "$name has neither src/lib.rs nor src/main.rs"; continue; }
    case "$name" in
        gpu | platform) ;;
        core)
            grep -q '^#!\[deny(unsafe_code)\]' "$root" || fail unsafe "$root must carry #![deny(unsafe_code)]"
            for f in $(find "$dir/src" -name '*.rs' | sort); do
                case "$f" in */alloc.rs | */alloc/*) continue ;; esac
                if grep -nE '\bunsafe\b' "$f" | grep -vE '^[0-9]+:[[:space:]]*//|unsafe_code' >/dev/null; then
                    fail unsafe "$f uses unsafe outside humanart-core::alloc"
                fi
            done
            ;;
        *) grep -q '^#!\[forbid(unsafe_code)\]' "$root" || fail unsafe "$root must carry #![forbid(unsafe_code)]" ;;
    esac
done
# Every unsafe block or impl states its invariant.
for f in $(rust_files); do
    awk -v file="$f" '
        { line[NR] = $0 }
        /^[[:space:]]*\/\// { next }
        /\bunsafe[[:space:]]*(\{|impl|fn)/ && $0 !~ /unsafe_code|unsafe_op_in_unsafe_fn/ {
            ok = 0
            for (i = NR - 6; i <= NR; i++) if (i > 0 && (line[i] ~ /SAFETY:/ || line[i] ~ /# Safety/)) ok = 1
            if (!ok) printf "FAIL unsafe: %s:%d: unsafe without a SAFETY comment\n", file, NR
        }' "$f"
done >"${TMPDIR:-/tmp}/humanart-check-unsafe.$$"
if [ -s "${TMPDIR:-/tmp}/humanart-check-unsafe.$$" ]; then
    cat "${TMPDIR:-/tmp}/humanart-check-unsafe.$$"
    FAILED=1
fi
rm -f "${TMPDIR:-/tmp}/humanart-check-unsafe.$$"

# --- 5. Every shader is referenced by the manifest [PLAT-04] --------------------------------------
if [ -d shaders ]; then
    for f in $(find shaders -type f -name '*.wgsl' | sort); do
        rel=${f#shaders/}
        if ! grep -rqF "\"$rel\"" crates/shaders/src 2>/dev/null; then
            fail shaders "$f is not listed in the humanart-shaders manifest"
        fi
    done
    for f in $(find shaders -type f -not -name '*.wgsl' -not -name '*.md' | sort); do
        fail shaders "$f: only .wgsl sources (and notes in .md) live under shaders/"
    done
fi

# --- 6. wasm-compatible crates stay off the OS [PLAT-03] ------------------------------------------
for name in core math io model doc gpu shaders render ui runtime; do
    [ -d "crates/$name/src" ] || continue
    for f in $(find "crates/$name/src" -name '*.rs' | sort); do
        # Test-only files, as in section 2 (the wasm cargo check catches a tests.rs left ungated).
        case "$f" in */tests/* | */benches/* | */tests.rs) continue ;; esac
        awk -v file="$f" '
            /^[[:space:]]*#\[cfg\(test\)\]/ { exit }
            /^[[:space:]]*\/\// { next }
            /std::fs|std::net|std::process|thread::spawn|Instant::now|SystemTime::now/ {
                if ($0 !~ /wasm-ok:/) printf "FAIL wasm: %s:%d: %s\n", file, NR, $0
            }' "$f"
    done
done >"${TMPDIR:-/tmp}/humanart-check-wasm.$$"
if [ -s "${TMPDIR:-/tmp}/humanart-check-wasm.$$" ]; then
    cat "${TMPDIR:-/tmp}/humanart-check-wasm.$$"
    FAILED=1
fi
rm -f "${TMPDIR:-/tmp}/humanart-check-wasm.$$"

# --- 7. Evidence files exist [0.3] ----------------------------------------------------------------
for f in PROGRESS.md docs/decisions.md docs/blockers.md docs/dependencies.md docs/reference-conventions.md \
    docs/perf-deviations.md docs/data-provenance.md docs/shader-notes.md; do
    [ -s "$f" ] || fail evidence "$f is missing or empty"
done

# --- 8. Cargo-backed checks: naga validation of every shader, registry consumption [REG-02] --------
if [ "$MODE" != "--static" ]; then
    cargo test -q -p humanart-shaders >/dev/null 2>&1 || fail shaders "cargo test -p humanart-shaders failed (naga validation)"
    cargo test -q -p humanart-doc registry >/dev/null 2>&1 || fail registry "cargo test -p humanart-doc registry failed"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "check-tree: FAILED"
    exit 1
fi
echo "check-tree: ok"
