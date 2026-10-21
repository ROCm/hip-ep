#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
# Build this worktree without changing shared source or install directories.
set -euo pipefail

SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${WORKSPACE:-/scratch/fhanuman/RCOmEP}
OUT=${NEMOTRON_BUILD_ROOT:-$SOURCE/.nemotron-perf}
IMAGE=${IMAGE:-hipdnn-ep-build:llvm22-restored-20260929}
DEPS=$WORKSPACE/build/onnx-hipdnn-ep/_deps
SDK=$WORKSPACE/build/onnx-hipdnn-ep/_therock
mkdir -p "$OUT/build" "$OUT/install" "$OUT/home" "$OUT/tmp" "$OUT/logs"
LOG=$OUT/logs/build-$(date -u +%Y%m%dT%H%M%SZ).log

docker run --rm --network none --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$WORKSPACE,target=$WORKSPACE,readonly" \
    --mount "type=bind,source=$OUT,target=$OUT" \
    -e "HOME=$OUT/home" -e "TMPDIR=$OUT/tmp" \
    --entrypoint bash "$IMAGE" -c '
set -euo pipefail
source_dir=$1
out=$2
deps=$3
sdk=$4
workspace=$5
cmake -S "$source_dir" -B "$out/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_MOCK_RUNTIME=OFF \
    -DBUILD_HIP_UNIT_TESTS=ON -DHIP_ARCHITECTURES=gfx1151 \
    -DCMAKE_INSTALL_PREFIX="$out/install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH="$workspace/llvm24-install;$workspace/prebuilt-local" \
    -DCMAKE_DISABLE_FIND_PACKAGE_onnxruntime=ON \
    -DLLVM_DIR="$workspace/llvm24-install/lib/cmake/llvm" \
    -DMLIR_DIR="$workspace/llvm24-install/lib/cmake/mlir" \
    -DCLANG_EXECUTABLE="$workspace/llvm24-install/bin/clang" \
    -DLLVM_LINK_EXECUTABLE="$workspace/llvm24-install/bin/llvm-link" \
    -DHIPDNN_CLANG_EXECUTABLE="$workspace/llvm24-install/bin/clang++" \
    -DTHEROCK_DIST="$sdk" -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_CPPTRACE="$deps/cpptrace-src" \
    -DFETCHCONTENT_SOURCE_DIR_GLOG="$deps/glog-src" \
    -DFETCHCONTENT_SOURCE_DIR_HASH-LIBRARY="$deps/hash-library-src" \
    -DFETCHCONTENT_SOURCE_DIR_LIBDWARF="$deps/libdwarf-src" \
    -DFETCHCONTENT_SOURCE_DIR_MICROSOFT.GSL="$deps/microsoft.gsl-src" \
    -DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME="$out/deps/onnxruntime-linux-x64-1.31.0" \
    -DFETCHCONTENT_SOURCE_DIR_ZSTD="$deps/zstd-src"
cmake --build "$out/build" --parallel 8
cmake --install "$out/build"
' bash "$SOURCE" "$OUT" "$DEPS" "$SDK" "$WORKSPACE" 2>&1 | tee "$LOG"
