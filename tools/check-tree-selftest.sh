#!/bin/sh
# Proves check-tree.sh and check-deps.sh fail on injected violations [AC-WP01-3].
# Builds throwaway trees under a temp folder; touches nothing in the real tree.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
BAD=0

# A minimal clean tree.
fresh() {
    rm -rf "$WORK/t"
    mkdir -p "$WORK/t/crates/math/src" "$WORK/t/crates/core/src" "$WORK/t/crates/shaders/src" \
        "$WORK/t/docs" "$WORK/t/shaders/lib" "$WORK/t/tools"
    printf '[workspace]\nmembers = ["crates/*"]\n' >"$WORK/t/Cargo.toml"
    printf '[package]\nname = "humanart-math"\n\n[dependencies]\nhumanart-core.workspace = true\n' >"$WORK/t/crates/math/Cargo.toml"
    printf '//! Math.\n#![forbid(unsafe_code)]\n\npub fn one() -> u32 {\n    1\n}\n' >"$WORK/t/crates/math/src/lib.rs"
    printf '[package]\nname = "humanart-core"\n\n[dependencies]\n' >"$WORK/t/crates/core/Cargo.toml"
    printf '//! Core.\n#![deny(unsafe_code)]\n' >"$WORK/t/crates/core/src/lib.rs"
    printf '[package]\nname = "humanart-shaders"\n\n[dependencies]\n' >"$WORK/t/crates/shaders/Cargo.toml"
    printf '//! Shaders.\n#![forbid(unsafe_code)]\npub const M: &[&str] = &["lib/math.wgsl"];\n' >"$WORK/t/crates/shaders/src/lib.rs"
    printf 'fn f() {}\n' >"$WORK/t/shaders/lib/math.wgsl"
    for f in PROGRESS.md docs/decisions.md docs/blockers.md docs/dependencies.md docs/reference-conventions.md \
        docs/perf-deviations.md docs/data-provenance.md docs/shader-notes.md; do
        echo x >"$WORK/t/$f"
    done
}

run() { ROOT="$WORK/t" sh "$HERE/check-tree.sh" --static >"$WORK/out" 2>&1; }

expect_pass() {
    if run; then echo "ok   clean tree passes"; else
        echo "BAD  clean tree fails:"
        cat "$WORK/out"
        BAD=1
    fi
}

# expect_fail <what> <check name that must be reported>
expect_fail() {
    if run; then
        echo "BAD  $1 was not caught"
        BAD=1
    elif grep -q "^FAIL $2" "$WORK/out"; then
        echo "ok   $1 is caught"
    else
        echo "BAD  $1 failed for another reason:"
        cat "$WORK/out"
        BAD=1
    fi
}

fresh
expect_pass

fresh
printf 'pub fn two() -> u32 {\n    todo!()\n}\n' >>"$WORK/t/crates/math/src/lib.rs"
expect_fail "todo!()" stubs

fresh
printf '// FIX''ME later\n' >>"$WORK/t/crates/math/src/lib.rs"
expect_fail "a FIX""ME comment" stubs

fresh
printf 'pub fn two(x: Option<u32>) -> u32 {\n    x.unwrap()\n}\n' >>"$WORK/t/crates/math/src/lib.rs"
expect_fail "unwrap() outside tests" unwrap

fresh
printf 'pub fn two(x: Option<u32>) -> u32 {\n    // INVARIANT: always set by the caller.\n    x.expect("set")\n}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {\n        Some(1).unwrap();\n    }\n}\n' >>"$WORK/t/crates/math/src/lib.rs"
if run; then echo "ok   documented invariant and test unwraps pass"; else
    echo "BAD  documented invariant or test unwrap was flagged:"
    cat "$WORK/out"
    BAD=1
fi

fresh
printf 'serde = "1"\n' >>"$WORK/t/crates/math/Cargo.toml"
expect_fail "a forbidden crate" deps

fresh
printf '//! Math.\n\npub fn one() -> u32 {\n    1\n}\n' >"$WORK/t/crates/math/src/lib.rs"
expect_fail "a missing forbid(unsafe_code)" unsafe

fresh
printf 'fn g() {}\n' >"$WORK/t/shaders/lib/orphan.wgsl"
expect_fail "an unreferenced shader" shaders

fresh
printf 'pub fn now() -> std::time::Instant {\n    std::time::Instant::now()\n}\n' >>"$WORK/t/crates/math/src/lib.rs"
expect_fail "an OS clock in a wasm crate" wasm

fresh
rm "$WORK/t/docs/blockers.md"
expect_fail "a missing evidence file" evidence

# check-deps.sh: a crate that the record does not list must fail.
printf '# Dependencies\n\n## Recorded crates\n\n```\nwgpu\n```\n' >"$WORK/deps.md"
if printf 'wgpu\nsneaky\n' | DEPS_LIST=- DEPS_DOC="$WORK/deps.md" sh "$HERE/check-deps.sh" >"$WORK/out" 2>&1; then
    echo "BAD  check-deps passed an unrecorded crate"
    BAD=1
elif grep -q 'sneaky' "$WORK/out"; then
    echo "ok   check-deps catches an unrecorded crate"
else
    echo "BAD  check-deps failed for another reason:"
    cat "$WORK/out"
    BAD=1
fi
if printf 'wgpu\n' | DEPS_LIST=- DEPS_DOC="$WORK/deps.md" sh "$HERE/check-deps.sh" >"$WORK/out" 2>&1; then
    echo "ok   check-deps passes a recorded tree"
else
    echo "BAD  check-deps fails a recorded tree:"
    cat "$WORK/out"
    BAD=1
fi

if [ "$BAD" -ne 0 ]; then
    echo "check-tree-selftest: FAILED"
    exit 1
fi
echo "check-tree-selftest: ok"
