#!/bin/bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PREBUILT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)/../prebuilt-local
SCRATCH_DIR="${PREBUILT_DIR}/.src"

# Pins kept in lockstep with .github/workflows/windows-build.yml.
# LLVM is built from this upstream llvm/llvm-project commit (== 22.1.0).
LLVM_COMMIT="4434dabb69916856b824f68a64b029c67175e532"
PROTO_TAG="v34.0"
FLATBUFFERS_TAG="v25.12.19"

mkdir -p "$PREBUILT_DIR" "$SCRATCH_DIR"

# /MT static CRT on Windows to match the prebuilt LLVM and the EP build.
MSVC_RT_FLAG=""
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) MSVC_RT_FLAG="-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" ;;
esac

CCACHE_FLAGS=""
if command -v ccache >/dev/null 2>&1; then
    CCACHE_FLAGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

# --- LLVM / MLIR / LLD -------------------------------------------------------
# Only mlir;lld + the X86 backend are needed: the project find_package()s
# LLVM/MLIR/LLD (no Clang) and links X86 codegen + lldCOFF/lldCommon.
if [ -f "$PREBUILT_DIR/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "Already built: LLVM/MLIR/LLD"
else
    echo "Building LLVM/MLIR/LLD from source (commit $LLVM_COMMIT) ..."
    LLVM_SRC="$SCRATCH_DIR/llvm-project"
    if [ ! -d "$LLVM_SRC/.git" ]; then
        git init "$LLVM_SRC"
        git -C "$LLVM_SRC" remote add origin https://github.com/llvm/llvm-project.git
    fi
    git -C "$LLVM_SRC" fetch --depth 1 origin "$LLVM_COMMIT"
    git -C "$LLVM_SRC" checkout FETCH_HEAD
    cmake -G Ninja -S "$LLVM_SRC/llvm" -B "$SCRATCH_DIR/llvm-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir;lld" \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_INSTALL_PREFIX="$PREBUILT_DIR" \
        -DLLVM_ENABLE_RTTI=OFF \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_INSTALL_UTILS=OFF \
        $MSVC_RT_FLAG $CCACHE_FLAGS
    cmake --build "$SCRATCH_DIR/llvm-build" --target install
fi

# --- protobuf (+ abseil) -----------------------------------------------------
# CMAKE_CXX_STANDARD=17 is REQUIRED on MSVC: abseil pins its installed
# options.h ABI from a configure-time `_MSVC_LANG >= 201703` probe. Without it
# MSVC defaults to C++14 and abseil pins own-string_view while the libs build
# under C++17 (std::string_view) -> unresolved symbols at EP link time.
if [ -f "$PREBUILT_DIR/lib/cmake/protobuf/protobuf-config.cmake" ]; then
    echo "Already built: protobuf"
else
    echo "Building protobuf $PROTO_TAG from source ..."
    PB_SRC="$SCRATCH_DIR/protobuf"
    rm -rf "$PB_SRC"
    git clone --depth 1 --branch "$PROTO_TAG" --recursive \
        https://github.com/protocolbuffers/protobuf.git "$PB_SRC"
    cmake -G Ninja -S "$PB_SRC" -B "$SCRATCH_DIR/protobuf-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREBUILT_DIR" \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_EXAMPLES=OFF \
        -Dprotobuf_WITH_ZLIB=OFF \
        -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        -Dprotobuf_INSTALL=ON \
        $MSVC_RT_FLAG $CCACHE_FLAGS
    cmake --build "$SCRATCH_DIR/protobuf-build" --target install
fi

# --- flatbuffers -------------------------------------------------------------
if [ -f "$PREBUILT_DIR/lib/cmake/flatbuffers/flatbuffers-config.cmake" ]; then
    echo "Already built: flatbuffers"
else
    echo "Building flatbuffers $FLATBUFFERS_TAG from source ..."
    FB_SRC="$SCRATCH_DIR/flatbuffers"
    rm -rf "$FB_SRC"
    git clone --depth 1 --branch "$FLATBUFFERS_TAG" \
        https://github.com/google/flatbuffers.git "$FB_SRC"
    cmake -G Ninja -S "$FB_SRC" -B "$SCRATCH_DIR/flatbuffers-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREBUILT_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_BUILD_FLATC=ON \
        -DFLATBUFFERS_BUILD_FLATLIB=ON \
        $MSVC_RT_FLAG $CCACHE_FLAGS
    cmake --build "$SCRATCH_DIR/flatbuffers-build" --target install
fi

echo ""
echo "Dependencies built at: $PREBUILT_DIR"
echo "CMake configs:"
ls "$PREBUILT_DIR/lib/cmake/"
