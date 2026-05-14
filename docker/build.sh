#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# In-container build script. Implements steps A.4-A.8 of docs/quick_start.md
# "Linux Support → A. Build from source on Linux".
#
# Layout matches the Windows quick_start.md "Directory Layout" section:
# everything is a sibling under <workspace>/, mirroring the upstream pattern
# so cmake invocations in docs (e.g. `cmake --install ../build/onnxruntime`)
# work verbatim from the project root.
#
#   <workspace>/                          (= dirname $SOURCE_DIR)
#   ├── onnx-hipdnn-ep/      (SOURCE_DIR — this project's source)
#   ├── onnxruntime/         (ORT_SRC — ORT source clone, kept on disk so
#   │                          incremental rebuilds reuse the .git tree)
#   ├── onnxruntime-genai/   (OGA_SRC — OGA source clone, sibling of ORT
#   │                          with the repo name; only used when BUILD_OGA=1)
#   ├── prebuilt-local/      (PREBUILT_DIR — installed deps: ORT, protobuf,
#   │                          flatbuffers, etc.)
#   ├── therock-dist/        (THEROCK_DIST_DIR — TheRock ROCm SDK, ~13 GB)
#   ├── build/               (BUILD_ROOT)
#   │   ├── onnx-hipdnn-ep/      (BUILD_DIR — this project's cmake build)
#   │   ├── onnxruntime/         (ORT_BUILD — ORT cmake build)
#   │   ├── onnxruntime-genai/   (OGA_BUILD — OGA cmake build)
#   │   └── scratch/             (SCRATCH_DIR — transient src+build for
#   │                              protobuf / flatbuffers)
#   ├── install/             (INSTALL_DIR — this project's install prefix)
#   └── docker/              (this script + run.sh)
#
# Each step is idempotent: it checks for the install marker and skips if
# already done. Re-running this script after a partial failure resumes from
# the failed step. Wipe a specific dir (e.g. prebuilt-local/, therock-dist/)
# to force that step to redo.
#
# Required env (passed by run.sh):
#   SOURCE_DIR        — bind-mounted path of onnx-hipdnn-ep checkout
#   HIP_ARCHITECTURES — gpu arch (e.g. gfx1151)
#
# Optional (all default to the layout above; override individually if needed):
#   PREBUILT_DIR, THEROCK_DIST_DIR, BUILD_ROOT, BUILD_DIR, INSTALL_DIR,
#   ORT_SRC, ORT_BUILD, OGA_SRC, OGA_BUILD, SCRATCH_DIR
#   ONNXRUNTIME_VERSION (default 1.25.1)
#   PROTOBUF_REF        (default v34.0)
#   FLATBUFFERS_REF     (default v25.12.19)
#   THEROCK_VERSION     (default therock-dist-linux-gfx1151-7.11.0)
#   BUILD_OGA           (default 0; 1 = also build OGA model_benchmark)
#   SKIP_LIT            (default 0; 1 = skip LIT tests)
#   FORCE_RECONFIGURE   (default 0; 1 = re-run cmake even if build/ exists)

set -euo pipefail

: "${SOURCE_DIR:?SOURCE_DIR not set — run.sh should pass it via -e}"
: "${HIP_ARCHITECTURES:=gfx1151}"
: "${ONNXRUNTIME_VERSION:=1.25.1}"
: "${PROTOBUF_REF:=v34.0}"
: "${FLATBUFFERS_REF:=v25.12.19}"
: "${THEROCK_VERSION:=therock-dist-linux-gfx1151-7.11.0}"
# OGA pin: this is the single source of truth for "which OGA SHA to build
# against" inside the container. CI forwards `OGA_REF` from
# .github/workflows/linux-build.yml's env block via docker/run.sh, so
# bumping the workflow env is enough to roll OGA forward. Local dev
# without an explicit override gets this hard-coded pin.
: "${OGA_REF:=2615301864b4d2397231d443865cb96111cc5fc2}"
: "${BUILD_OGA:=0}"
: "${SKIP_LIT:=0}"
: "${FORCE_RECONFIGURE:=0}"

WORKSPACE="$(dirname "$SOURCE_DIR")"
PROJECT_NAME="$(basename "$SOURCE_DIR")"

: "${PREBUILT_DIR:=$WORKSPACE/prebuilt-local}"
: "${THEROCK_DIST_DIR:=$WORKSPACE/therock-dist}"
: "${BUILD_ROOT:=$WORKSPACE/build}"
: "${SCRATCH_DIR:=$BUILD_ROOT/scratch}"
: "${ORT_SRC:=$WORKSPACE/onnxruntime}"
: "${ORT_BUILD:=$BUILD_ROOT/onnxruntime}"
: "${OGA_SRC:=$WORKSPACE/onnxruntime-genai}"
: "${OGA_BUILD:=$BUILD_ROOT/onnxruntime-genai}"
: "${BUILD_DIR:=$BUILD_ROOT/$PROJECT_NAME}"
: "${INSTALL_DIR:=$WORKSPACE/install}"

mkdir -p "$PREBUILT_DIR" "$THEROCK_DIST_DIR" "$BUILD_ROOT" "$SCRATCH_DIR"

NPROC=$(nproc)
step() {
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "▶ $*"
    echo "════════════════════════════════════════════════════════════════════"
}

step "Build configuration"
cat <<EOF
WORKSPACE         = $WORKSPACE
SOURCE_DIR        = $SOURCE_DIR
ORT_SRC           = $ORT_SRC
PREBUILT_DIR      = $PREBUILT_DIR
THEROCK_DIST_DIR  = $THEROCK_DIST_DIR
BUILD_ROOT        = $BUILD_ROOT
BUILD_DIR         = $BUILD_DIR
ORT_BUILD         = $ORT_BUILD
SCRATCH_DIR       = $SCRATCH_DIR
INSTALL_DIR       = $INSTALL_DIR
HIP_ARCHITECTURES = $HIP_ARCHITECTURES
ONNXRUNTIME_VERSION = $ONNXRUNTIME_VERSION
PROTOBUF_REF      = $PROTOBUF_REF
FLATBUFFERS_REF   = $FLATBUFFERS_REF
THEROCK_VERSION   = $THEROCK_VERSION
BUILD_OGA         = $BUILD_OGA
SKIP_LIT          = $SKIP_LIT
nproc             = $NPROC
EOF

# Quick-checking functions: each returns 0 if the install is already valid.
have_therock() {
    [ -d "$THEROCK_DIST_DIR/lib/llvm/bin" ] || [ -f "$THEROCK_DIST_DIR/bin/amdgpu-arch" ]
}
have_ort() {
    ls "$PREBUILT_DIR"/lib/cmake/onnxruntime/onnxruntimeConfig.cmake >/dev/null 2>&1 \
        || ls "$PREBUILT_DIR"/lib/cmake/onnxruntime/onnxruntime-config.cmake >/dev/null 2>&1
}
have_protobuf() {
    ls "$PREBUILT_DIR"/lib/cmake/protobuf/protobuf-config.cmake >/dev/null 2>&1 \
        || ls "$PREBUILT_DIR"/lib/cmake/utf8_range/utf8_rangeConfig.cmake >/dev/null 2>&1
}
have_flatbuffers() {
    # flatbuffers v25.12.19 installs the cmake package config as
    # `flatbuffers-config.cmake` (lowercase, hyphenated), not the
    # CamelCase `FlatBuffersConfig.cmake` an earlier flatbuffers release
    # used. Both names are valid per find_package(flatbuffers) semantics,
    # but the actual file determines whether we can skip A.6b. Probing
    # the wrong name made A.6b rebuild on every CI run even after a
    # prebuilt-local cache hit (~2-3 min wasted each time).
    ls "$PREBUILT_DIR"/lib/cmake/flatbuffers/flatbuffers-config.cmake >/dev/null 2>&1 \
        || ls "$PREBUILT_DIR"/lib/cmake/flatbuffers/FlatBuffersConfig.cmake >/dev/null 2>&1
}
have_oga() {
    # OGA produces two install-facing artifacts that downstream consumers
    # (gpu-perf-accuracy-test.yml's model_benchmark runs, this artifact's
    # bin/lib trees) actually use: the model_benchmark CLI binary and the
    # libonnxruntime-genai.so* SONAME chain. When both are in place we
    # can skip A.9 entirely. The CI's `Cache OGA install outputs` step
    # restores exactly these paths on cache hit, so this check is what
    # bridges the GHA cache to the skip-build path.
    [ -x "$INSTALL_DIR/bin/model_benchmark" ] \
        && ls "$INSTALL_DIR"/lib/libonnxruntime-genai.so* >/dev/null 2>&1
}

# A.4 — TheRock SDK
if have_therock; then
    echo "[skip] TheRock already at $THEROCK_DIST_DIR"
else
    step "A.4  Download TheRock $THEROCK_VERSION"
    TROCK_TAR="$SCRATCH_DIR/therock.tar.gz"
    curl -fsSL "https://repo.amd.com/rocm/tarball/${THEROCK_VERSION}.tar.gz" -o "$TROCK_TAR"
    tar -xzf "$TROCK_TAR" -C "$THEROCK_DIST_DIR" --strip-components=1
    rm -f "$TROCK_TAR"
fi

# A.5 — ONNX Runtime from source
if have_ort; then
    echo "[skip] ONNX Runtime already installed in $PREBUILT_DIR"
else
    step "A.5  Build ONNX Runtime v${ONNXRUNTIME_VERSION} (50-90 min cold)"
    if [ ! -d "$ORT_SRC/.git" ]; then
        rm -rf "$ORT_SRC"
        git clone --depth 1 --recursive --branch "v${ONNXRUNTIME_VERSION}" \
            https://github.com/Microsoft/onnxruntime.git "$ORT_SRC"
    else
        echo "[skip] $ORT_SRC/.git exists — using user-managed checkout as-is"
    fi
    cd "$ORT_SRC"
    # NOTE: docs/quick_start.md A.5 lists --disable_memleak_checker, but ORT
    # 1.25.1's Linux build.py rejects it (it's a Windows-only flag from the
    # build.bat path). Dropping it; memleak checker only matters for tests
    # which we --skip_tests anyway.
    # No --build_wheel: it pulls in onnxruntime_python.cmake which needs
    # Python::NumPy and python dev headers. The hipdnn-ep build only needs
    # the C++ libs + headers + onnxruntime_perf_test.
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel "$NPROC" \
        --compile_no_warning_as_error \
        --skip_submodule_sync \
        --build_dir "$ORT_BUILD" \
        --skip_tests \
        --cmake_generator Ninja
    # ORT 1.25.1 Linux build.sh writes to $build_dir/Release (no Linux/ prefix
    # despite docs A.5 implying otherwise — that's the Windows layout).
    cmake --install "$ORT_BUILD/Release" --prefix "$PREBUILT_DIR"
    PERF_TEST=$(find "$ORT_BUILD" -name onnxruntime_perf_test -type f | head -1)
    mkdir -p "$PREBUILT_DIR/bin"
    if [ -n "$PERF_TEST" ]; then cp "$PERF_TEST" "$PREBUILT_DIR/bin/"; fi
fi

# OGA's cmake/global_variables.cmake (A.9 below) checks for the ORT C API
# header at $ORT_HOME/include/onnxruntime_c_api.h (flat), but ORT 1.25.1's
# `cmake --install` puts them under $PREBUILT_DIR/include/onnxruntime/*.h
# (nested). Mirror the public headers up to include/ so both layouts
# resolve.
#
# This runs unconditionally — NOT inside the A.5 `if have_ort` block —
# because actions/cache may restore a prebuilt-local/ from a previous
# run that didn't have the flatten step, so flattening only on first
# fresh install would leave cache-hit runs broken. cp -n keeps it
# idempotent: any file already at include/ root (incl. those from
# protobuf / flatbuffers installs) is never clobbered.
if [ -d "$PREBUILT_DIR/include/onnxruntime" ]; then
    cp -rn "$PREBUILT_DIR/include/onnxruntime/." "$PREBUILT_DIR/include/" 2>/dev/null || true
fi

# A.6a — protobuf v34
if have_protobuf; then
    echo "[skip] protobuf already installed in $PREBUILT_DIR"
else
    step "A.6  Build protobuf $PROTOBUF_REF"
    PB_SRC="$SCRATCH_DIR/protobuf"
    PB_BUILD="$SCRATCH_DIR/protobuf-build"
    rm -rf "$PB_SRC" "$PB_BUILD"
    git clone --depth 1 --branch "$PROTOBUF_REF" --recursive \
        https://github.com/protocolbuffers/protobuf.git "$PB_SRC"
    cmake -G Ninja -B "$PB_BUILD" -S "$PB_SRC" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREBUILT_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_EXAMPLES=OFF \
        -Dprotobuf_WITH_ZLIB=OFF \
        -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        -Dprotobuf_INSTALL=ON
    ninja -C "$PB_BUILD" install
    rm -rf "$PB_SRC" "$PB_BUILD"
fi

# A.6b — flatbuffers v25
if have_flatbuffers; then
    echo "[skip] flatbuffers already installed in $PREBUILT_DIR"
else
    step "A.6  Build flatbuffers $FLATBUFFERS_REF"
    FB_SRC="$SCRATCH_DIR/flatbuffers"
    FB_BUILD="$SCRATCH_DIR/flatbuffers-build"
    rm -rf "$FB_SRC" "$FB_BUILD"
    git clone --depth 1 --branch "$FLATBUFFERS_REF" \
        https://github.com/google/flatbuffers.git "$FB_SRC"
    cmake -G Ninja -B "$FB_BUILD" -S "$FB_SRC" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREBUILT_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_BUILD_FLATC=ON \
        -DFLATBUFFERS_BUILD_FLATLIB=ON
    ninja -C "$FB_BUILD" install
    rm -rf "$FB_SRC" "$FB_BUILD"
fi

# A.2 — submodules. Run on the HOST first if any submodule is private (the
# container has no git creds). With submodules already checked out, this is
# a no-op verification pass. Note: NOT --depth 1 — MorphiZen's recorded
# commit is not at branch tip, so shallow fetch fails.
step "A.2  Verify submodules (incl. MorphiZen)"
cd "$SOURCE_DIR"
# git on host vs container can disagree on the .git working-tree owner when
# UID matches but the namespace differs; mark this tree safe. Also whitelist
# the submodule and sibling source dirs that A.5 (ORT) and A.9 (OGA) touch.
# We deliberately do NOT use `safe.directory '*'` — that swallows owner
# mismatches everywhere and can mask real environment bugs.
git config --global --add safe.directory "$SOURCE_DIR"
git config --global --add safe.directory "$SOURCE_DIR/3rd-party/morphizen"
git config --global --add safe.directory "$ORT_SRC"
git config --global --add safe.directory "$OGA_SRC"
# Trust user-managed submodule checkouts (same .git-guard pattern as A.5 ORT
# and A.9 OGA below). The host normally runs
# `git submodule update --init --recursive` with ssh-agent loaded before
# invoking docker/run.sh, and CI does the same via webfactory/ssh-agent in
# linux-build.yml. The container does NOT carry SSH credentials for the
# private ROCm/MorphiZen repo, so we cannot run `submodule update` here;
# but we also do not need to, because the submodule is already on disk via
# the bind-mounted workspace.
if [ -f 3rd-party/morphizen/CMakeLists.txt ]; then
    echo "[skip] 3rd-party/morphizen already populated — trusting user-managed checkout"
else
    git submodule update --init --recursive
fi

# Optional sccache integration. mozilla-actions/sccache-action exports
# SCCACHE_GHA_ENABLED + ACTIONS_CACHE_URL + ACTIONS_RUNTIME_TOKEN +
# ACTIONS_RESULTS_URL (cache v2) into the runner env; docker/run.sh
# forwards all four into the container. Wrap compilers with sccache only
# when ALL of these are present:
#   1. SCCACHE_GHA_ENABLED=true
#   2. sccache binary on PATH (Dockerfile Layer 4 installs it)
#   3. ACTIONS_CACHE_URL or ACTIONS_RESULTS_URL non-empty
# The third check is the gate that broke CI run 25804869662: the
# sccache-action exports SCCACHE_GHA_ENABLED unconditionally, but the
# cache URL only when GitHub's cache service is actually reachable from
# the runner. Without it, sccache --start-server fails immediately and
# every ninja compile job dies with "ACTIONS_CACHE_URL not found".
#
# Outside CI all three are empty, so SCCACHE_LAUNCHER_ARGS stays empty
# and the build runs without a compiler launcher (uncached, as expected
# for local dev).
SCCACHE_LAUNCHER_ARGS=()
if [ "${SCCACHE_GHA_ENABLED:-}" = "true" ] \
        && command -v sccache >/dev/null 2>&1 \
        && [ -n "${ACTIONS_CACHE_URL:-}${ACTIONS_RESULTS_URL:-}" ]; then
    SCCACHE_LAUNCHER_ARGS=(
        -DCMAKE_C_COMPILER_LAUNCHER=sccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    )
    echo "[sccache] enabled (gha backend; cache URL detected)"
else
    echo "[sccache] disabled — no GHA cache URL in env (cold build expected)"
fi

# A.7 — Configure + build hipdnn-ep. BUILD_DIR / INSTALL_DIR were resolved
# at the top of this script (defaults: <workspace>/build/<repo>/ and
# <workspace>/install/). Source tree stays untouched.
if [ "$FORCE_RECONFIGURE" = "1" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    step "A.7  Configure (HIP_ARCHITECTURES=$HIP_ARCHITECTURES)"
    cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
        -G Ninja \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_PREFIX_PATH=$PREBUILT_DIR;$THEROCK_DIST_DIR;/usr/lib/llvm-22" \
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        "-DTHEROCK_DIST=$THEROCK_DIST_DIR" \
        -DHIP_PLATFORM=amd \
        "-DHIP_ARCHITECTURES=$HIP_ARCHITECTURES" \
        -DBUILD_EP=ON \
        -DBUILD_MOCK_RUNTIME=OFF \
        -DBUILD_HIP_TOOLS=ON \
        -Dmorphizen_ENABLE_UNIT_TEST=OFF \
        "${SCCACHE_LAUNCHER_ARGS[@]}"
else
    echo "[skip] $BUILD_DIR/CMakeCache.txt exists — skipping configure (FORCE_RECONFIGURE=1 to redo)"
fi

step "A.7  Build hipdnn-ep"
cmake --build "$BUILD_DIR" --parallel "$NPROC"

step "A.7  Install -> $INSTALL_DIR"
cmake --install "$BUILD_DIR"

# A.7b — Stage runtime .so deps into install/lib so the install/ tree is
# self-contained (LD_LIBRARY_PATH=install/lib suffices to run hip-onnx-runner,
# onnxruntime_perf_test, model_benchmark, ...). Without this step, install/lib
# only carries libhip-compiler.so + libonnxruntime_morphizen_ep.so and the
# caller still has to add prebuilt-local/lib (ORT) and therock-dist/lib (HIP)
# to LD_LIBRARY_PATH.
#
# Implementation:
#   Pass 1 — ldd the EP + CLI tools with the dep prefixes on LD_LIBRARY_PATH
#            so SONAMEs without RPATH still resolve; pick up every .so that
#            lives under THEROCK_DIST_DIR or PREBUILT_DIR and copy it (plus
#            its readlink target so the SONAME chain stays valid) into
#            install/lib.
#   Pass 2 — ldd the just-staged set so transitive deps (e.g. libamdhip64 ->
#            librocprofiler-register, libhsa-runtime64 -> librocm_sysdeps_*)
#            are caught.
# We deliberately do NOT copy /usr/lib/x86_64-linux-gnu/libLLVM.so.22.1 et al.
# — those are container apt packages on the default ld.so search path; they
# stay system-managed.
step "A.7b  Stage runtime .so into install/lib"
stage_from_ldd_pass() {
    LD_LIBRARY_PATH="$THEROCK_DIST_DIR/lib:$PREBUILT_DIR/lib:$INSTALL_DIR/lib" \
        ldd "$@" 2>/dev/null \
        | awk -v t="$THEROCK_DIST_DIR" -v p="$PREBUILT_DIR" \
              '$3 ~ t || $3 ~ p {print $3}' \
        | sort -u \
        | while read -r src; do
            dst="$INSTALL_DIR/lib/$(basename "$src")"
            if [ ! -e "$dst" ]; then
                cp -a "$src" "$dst"
                # If $src is a symlink, also copy the readlink target so the
                # `lib<name>.so.1 -> lib<name>.so.1.X.Y` chain resolves.
                real=$(readlink -f "$src")
                rb=$(basename "$real")
                if [ "$rb" != "$(basename "$src")" ] && [ ! -e "$INSTALL_DIR/lib/$rb" ]; then
                    cp -a "$real" "$INSTALL_DIR/lib/"
                fi
            fi
        done
}

# Pass 1: direct deps of the EP + CLI tools.
stage_from_ldd_pass \
    "$INSTALL_DIR/lib/libonnxruntime_morphizen_ep.so" \
    "$INSTALL_DIR/lib/libhip-compiler.so" \
    "$INSTALL_DIR/bin/hip-compiler" \
    "$INSTALL_DIR/bin/hip-onnx-runner" \
    "$INSTALL_DIR/bin/hip-test-dll"

# Pass 2: transitive — ldd the libs we just staged. Loop to a fixed point so
# any second-level deps that Pass 2 itself introduces (rare with current ORT
# + TheRock SO graph, but cheap to be defensive) are also picked up.
prev_count=0
for _i in 1 2 3 4; do
    mapfile -t _staged < <(find "$INSTALL_DIR/lib" -maxdepth 1 -name 'lib*.so*' -type f)
    [ "${#_staged[@]}" -eq 0 ] && break
    stage_from_ldd_pass "${_staged[@]}"
    new_count=$(find "$INSTALL_DIR/lib" -maxdepth 1 -name 'lib*.so*' -type f | wc -l)
    if [ "$new_count" -eq "$prev_count" ]; then break; fi
    prev_count=$new_count
done
echo "[stage] $(find "$INSTALL_DIR/lib" -name 'lib*.so*' | wc -l) lib*.so* in $INSTALL_DIR/lib/"

# A.8 — LIT tests (no GPU needed)
if [ "$SKIP_LIT" != "1" ]; then
    step "A.8  Run LIT tests"
    cmake --build "$BUILD_DIR" --target check-hip-mlir-lit
fi

# A.9 — OGA model_benchmark (optional)
#
# Source layout mirrors ORT: <workspace>/onnxruntime-genai/ (sibling to
# onnxruntime/, named after the repo) so users who manage their own checkout
# can drop it in beside ORT without touching env vars. The .git guard below
# means an existing checkout is respected as-is — no fetch, no checkout, no
# submodule update. OGA_REF (resolved at the top of this script from env or
# the hard-coded default) is applied only to a fresh clone.
#
# Skip-the-build gate: have_oga() returns true when the install artifacts
# are already in place — either built earlier this run, or restored from
# CI's `Cache OGA install outputs` cache step. The CI cache key includes
# OGA_REF + ONNXRUNTIME_VERSION, so bumping either at the workflow level
# automatically misses the cache here and triggers a rebuild.
if [ "$BUILD_OGA" = "1" ]; then
    if have_oga; then
        echo "[skip] OGA model_benchmark + libonnxruntime-genai.so* already in $INSTALL_DIR"
    else
        step "A.9  (optional) Build OGA model_benchmark"
        if [ ! -d "$OGA_SRC/.git" ]; then
            rm -rf "$OGA_SRC"
            git clone --recursive https://github.com/AMDmoore/onnxruntime-genai.git "$OGA_SRC"
            ( cd "$OGA_SRC" && git checkout "$OGA_REF" && git submodule update --init --recursive )
        else
            echo "[skip] $OGA_SRC/.git exists — using user-managed checkout as-is"
        fi
        # --skip_wheel: the OGA python wheel is unusable on Linux without
        # an ORT python wheel (the OGA wheel does `import onnxruntime` at
        # runtime), and A.5 deliberately omits ORT --build_wheel to avoid
        # the Python::NumPy / dev-headers dependency drag. Skipping the
        # OGA wheel saves ~1-3 min of pybind11 + pip wheel pack on cold
        # builds. Consumers who need the wheel build OGA from source.
        python3 "$OGA_SRC/build.py" \
            --config Release \
            --cmake_generator Ninja \
            --ort_home "$PREBUILT_DIR" \
            --skip_tests --skip_examples --skip_wheel \
            --parallel \
            --build_dir "$OGA_BUILD"
        cp "$OGA_BUILD/Release/benchmark/c/model_benchmark" "$INSTALL_DIR/bin/"
        cp -a "$OGA_BUILD"/Release/libonnxruntime-genai.so* "$INSTALL_DIR/lib/"
    fi
    # OGA brought in libonnxruntime-genai.so (and possibly transitively
    # libonnxruntime.so already staged in A.7b). Re-run one ldd pass so
    # any new deps OGA pulls in (e.g. pybind11 runtime, tokenizers) land
    # in install/lib too. Runs regardless of build-vs-skip so a cache-
    # restored libonnxruntime-genai.so still drives its transitive .so
    # set through the closure check below.
    stage_from_ldd_pass \
        "$INSTALL_DIR/bin/model_benchmark" \
        "$INSTALL_DIR"/lib/libonnxruntime-genai.so*
fi

# Final closure check: with ONLY install/lib on LD_LIBRARY_PATH, every binary
# under install/bin and every .so under install/lib must resolve all of its
# DT_NEEDED entries. Runs AFTER A.9 so OGA's deps are covered.
#
# Failure here means the artifact will break on a fresh host (where the
# build-host's apt LLVM / TheRock / ORT prefixes are absent) — better to
# surface it now than ship a broken linux-gpu-test-package.
step "Verify install/ closure (LD_LIBRARY_PATH=install/lib)"
verify_install_closure() {
    local fail=0
    local target unresolved
    while IFS= read -r target; do
        # ldd a non-ELF file (e.g. shell wrapper script we accidentally
        # dropped into bin/) prints "not a dynamic executable" — skip those.
        unresolved=$(LD_LIBRARY_PATH="$INSTALL_DIR/lib" ldd "$target" 2>&1 \
                     | grep -F "not found" || true)
        if [ -n "$unresolved" ]; then
            echo "ERROR: unresolved deps in $target:" >&2
            echo "$unresolved" >&2
            fail=1
        fi
    done < <(
        find "$INSTALL_DIR/bin" -maxdepth 1 -type f -executable 2>/dev/null
        find "$INSTALL_DIR/lib" -maxdepth 1 -name 'lib*.so*' -type f 2>/dev/null
    )
    return $fail
}
if ! verify_install_closure; then
    echo "ERROR: install/ is not self-contained — see unresolved deps above." >&2
    exit 1
fi
echo "[verify] install/ closure OK"

step "DONE"
echo "Install tree: $INSTALL_DIR"
echo
echo "bin/:"
ls -la "$INSTALL_DIR/bin" 2>/dev/null | head -40 || true
echo
echo "lib/ (selected):"
ls -la "$INSTALL_DIR/lib" 2>/dev/null | grep -E '\.(so|so\.|a)$|libhip|libonnxruntime' | head -20 || true
