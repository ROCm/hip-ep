#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# In-container build script. Each step is idempotent (skips if its install
# marker is present); wipe a dir to force that step to redo.
#
# Sibling-of-source layout under <workspace>/ (= dirname $SOURCE_DIR):
#   onnx-hipdnn-ep/      SOURCE_DIR
#   onnxruntime/         ORT source     onnxruntime-genai/  OGA source (BUILD_OGA=1)
#   prebuilt-local/      built deps     therock-dist/       ROCm SDK
#   build/               cmake builds   install/            install prefix
#
# Required env (passed by run.sh): SOURCE_DIR, HIP_ARCHITECTURES.
# Knobs (defaulted below): ONNXRUNTIME_VERSION, PROTOBUF_REF, FLATBUFFERS_REF,
# THEROCK_VERSION, OGA_REF, BUILD_OGA, SKIP_LIT, FORCE_RECONFIGURE.

set -euo pipefail

: "${SOURCE_DIR:?SOURCE_DIR not set — run.sh should pass it via -e}"
: "${HIP_ARCHITECTURES:=gfx1151}"
: "${ONNXRUNTIME_VERSION:=1.25.1}"
# Space-separated list of microsoft/onnxruntime PR numbers to fetch from
# github.com/microsoft/onnxruntime/pull/<n>.patch and `git apply` on top
# of the pinned ONNXRUNTIME_VERSION tag in A.5. Remove a PR number once
# its fix lands in the pinned ONNXRUNTIME_VERSION release. Kept in
# lockstep with .github/workflows/linux-build.yml's env block; CI
# forwards the workflow value via docker/run.sh.
: "${ONNXRUNTIME_PR_PATCHES:=28608}"
: "${PROTOBUF_REF:=v34.0}"
: "${FLATBUFFERS_REF:=v25.12.19}"
: "${THEROCK_VERSION:=therock-dist-linux-gfx1151-7.11.0}"
# OGA pin (single source of truth; CI overrides via linux-build.yml env).
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
    # v25.12.19 ships lowercase `flatbuffers-config.cmake`; older releases used
    # CamelCase. Probe both so A.6b actually skips on cache hit.
    ls "$PREBUILT_DIR"/lib/cmake/flatbuffers/flatbuffers-config.cmake >/dev/null 2>&1 \
        || ls "$PREBUILT_DIR"/lib/cmake/flatbuffers/FlatBuffersConfig.cmake >/dev/null 2>&1
}
have_oga() {
    # Bridges the CI `Cache OGA install outputs` to A.9's skip path.
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

    # Fetch + apply microsoft/onnxruntime PR patches listed in
    # ONNXRUNTIME_PR_PATCHES. Each `.patch` endpoint returns the git
    # format-patch mailbox for the PR's current head. Idempotent: a patch
    # whose changes are already in the working tree (cache-restored ORT
    # clone from a previous run with the same PR set) is detected via
    # `git apply --reverse --check` and skipped; a patch that neither
    # applies forward nor reverses cleanly hard-fails. CI cache keys for
    # both `prebuilt-local` and the ORT source clone embed
    # ONNXRUNTIME_PR_PATCHES, so changing the PR set forces a fresh clone
    # path that always applies cleanly.
    cd "$ORT_SRC"
    for pr in $ONNXRUNTIME_PR_PATCHES; do
        [ -n "$pr" ] || continue
        url="https://github.com/microsoft/onnxruntime/pull/${pr}.patch"
        tmp="$SCRATCH_DIR/ort-pr-${pr}.patch"
        echo "[fetch] $url"
        curl -fsSL "$url" -o "$tmp"
        if git apply --check --reverse "$tmp" >/dev/null 2>&1; then
            echo "[skip-patch] PR #${pr} (already applied)"
        elif git apply --check "$tmp" >/dev/null 2>&1; then
            echo "[apply-patch] PR #${pr}"
            git apply --whitespace=nowarn "$tmp"
        else
            echo "[error] PR #${pr} neither applies nor is already applied" >&2
            exit 1
        fi
    done
    cd "$ORT_SRC"
    # No --build_wheel: ORT's python wheel target pulls Python::NumPy and
    # dev headers; we only need C++ libs + headers + onnxruntime_perf_test.
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel "$NPROC" \
        --compile_no_warning_as_error \
        --skip_submodule_sync \
        --build_dir "$ORT_BUILD" \
        --skip_tests \
        --cmake_generator Ninja
    # ORT 1.25.1 Linux build.sh writes to $build_dir/Release (no Linux/
    # prefix; that's the Windows layout).
    cmake --install "$ORT_BUILD/Release" --prefix "$PREBUILT_DIR"
    PERF_TEST=$(find "$ORT_BUILD" -name onnxruntime_perf_test -type f | head -1)
    mkdir -p "$PREBUILT_DIR/bin"
    if [ -n "$PERF_TEST" ]; then cp "$PERF_TEST" "$PREBUILT_DIR/bin/"; fi
fi

# OGA expects ORT C API headers flat at $ORT_HOME/include/onnxruntime_c_api.h
# but ORT installs them nested under include/onnxruntime/. Flatten so both
# layouts resolve. Runs outside the `if have_ort` block so cache-restored
# prebuilt-local/ from older runs also gets the mirror; cp -n is idempotent.
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

# A.2 — submodules. The container has no SSH creds for the private MorphiZen
# repo; host runs `git submodule update --init --recursive` with ssh-agent
# before invoking us (CI does the same via webfactory/ssh-agent). Whitelist
# only the dirs we touch — avoid `safe.directory '*'` which masks real owner
# bugs.
step "A.2  Verify submodules (incl. MorphiZen)"
cd "$SOURCE_DIR"
git config --global --add safe.directory "$SOURCE_DIR"
git config --global --add safe.directory "$SOURCE_DIR/3rd-party/morphizen"
git config --global --add safe.directory "$ORT_SRC"
git config --global --add safe.directory "$OGA_SRC"
if [ -f 3rd-party/morphizen/CMakeLists.txt ]; then
    echo "[skip] 3rd-party/morphizen already populated — trusting user-managed checkout"
else
    git submodule update --init --recursive
fi

# A.7 — Configure + build hipdnn-ep. Out-of-source: source tree untouched.
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
        -Dmorphizen_ENABLE_UNIT_TEST=OFF
else
    echo "[skip] $BUILD_DIR/CMakeCache.txt exists — skipping configure (FORCE_RECONFIGURE=1 to redo)"
fi

step "A.7  Build hipdnn-ep"
cmake --build "$BUILD_DIR" --parallel "$NPROC"

step "A.7  Install -> $INSTALL_DIR"
cmake --install "$BUILD_DIR"

# A.7b — Stage prebuilt-local deps (ORT + protobuf + flatbuffers) into
# install/lib so LD_LIBRARY_PATH=install/lib:therock-dist/lib is enough.
#
# We do NOT bundle TheRock-origin .so files (libamdhip64, libhsa-runtime64,
# librocm_sysdeps_*, libamd_comgr_loader, ...). Matches the Windows
# artifact contract (the windows-build.yml gpu-test job downloads TheRock
# separately on the GPU host). Reasons:
#   1. Halves artifact size (310 -> 150 MB).
#   2. libamd_comgr_loader.so is a stub that dlopens libamd_comgr.so.3 at
#      runtime — ldd misses the dlopen, so we'd ship the stub without its
#      target and silently break consumers without TheRock on PATH.
#   3. ROCm userland shares a private ioctl ABI with the amdgpu kernel
#      driver; bundling a fixed libamdhip64 risks kernel-vs-userland skew.
#
# 2-pass ldd: pass 1 ldds the EP + CLI tools; pass 2 loops over the just-
# staged libs to catch transitive deps (e.g. onnxruntime_providers_shared.so).
# awk filter copies only PREBUILT_DIR-resolved paths; TheRock + /usr/lib
# entries stay system-managed.
step "A.7b  Stage runtime .so into install/lib (prebuilt-local only)"
stage_from_ldd_pass() {
    LD_LIBRARY_PATH="$THEROCK_DIST_DIR/lib:$PREBUILT_DIR/lib:$INSTALL_DIR/lib" \
        ldd "$@" 2>/dev/null \
        | awk -v p="$PREBUILT_DIR" '$3 ~ p {print $3}' \
        | sort -u \
        | while read -r src; do
            dst="$INSTALL_DIR/lib/$(basename "$src")"
            if [ ! -e "$dst" ]; then
                cp -a "$src" "$dst"
                # Also copy the readlink target so the SONAME chain resolves.
                real=$(readlink -f "$src")
                rb=$(basename "$real")
                if [ "$rb" != "$(basename "$src")" ] && [ ! -e "$INSTALL_DIR/lib/$rb" ]; then
                    cp -a "$real" "$INSTALL_DIR/lib/"
                fi
            fi
        done
}

stage_from_ldd_pass \
    "$INSTALL_DIR/lib/libonnxruntime_morphizen_ep.so" \
    "$INSTALL_DIR/lib/libhip-compiler.so" \
    "$INSTALL_DIR/bin/hip-compiler" \
    "$INSTALL_DIR/bin/hip-onnx-runner" \
    "$INSTALL_DIR/bin/hip-test-dll"

# Loop to a fixed point so second-level transitive deps are caught too.
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

# A.9 — OGA model_benchmark (optional). have_oga skips the build when the
# install artifacts are present (either built this run or restored from CI's
# `Cache OGA install outputs`). OGA_REF bumps invalidate that cache key.
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
        # --skip_wheel: the OGA python wheel needs an ORT python wheel at
        # `import` time, but A.5 omits ORT --build_wheel; shipping it would
        # be useless and burns ~1-3 min on pybind11 + wheel pack.
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
    # Drag OGA's transitive prebuilt-local deps (pybind11 etc.) into
    # install/lib. Runs in both build + skip branches so cache-restored
    # libonnxruntime-genai.so still feeds the closure check below.
    stage_from_ldd_pass \
        "$INSTALL_DIR/bin/model_benchmark" \
        "$INSTALL_DIR"/lib/libonnxruntime-genai.so*
fi

# Closure check: with install/lib + therock-dist/lib on LD_LIBRARY_PATH
# (the supported runtime invocation, per docs/quick_start_linux.md), every
# install/bin binary and install/lib .so must resolve all DT_NEEDED. Catches
# leaks into install/lib that depend on a prefix outside {install,
# therock-dist, system-apt} before they ship.
step "Verify install/ closure (LD_LIBRARY_PATH=install/lib:therock-dist/lib)"
verify_install_closure() {
    local fail=0
    local target unresolved
    while IFS= read -r target; do
        unresolved=$(LD_LIBRARY_PATH="$INSTALL_DIR/lib:$THEROCK_DIST_DIR/lib" \
                     ldd "$target" 2>&1 \
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
