#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -euo pipefail

sha=$1
variant=$2
[[ $sha =~ ^[0-9a-f]{40}$ ]]
targets=X86
components="clang;clang-resource-headers;llvm-link;llvm-config;llvm-tblgen;mlir-tblgen;mlir-pdll;mlir-linalg-ods-yaml-gen;mlir-opt;lld;FileCheck;not;count;split-file;llvm-headers;llvm-libraries;cmake-exports;mlir-headers;mlir-libraries;mlir-cmake-exports;lld-headers;lld-libraries;lld-cmake-exports"
case "$variant" in
  x86) ;;
  x86-amdgpu)
    targets='X86;AMDGPU'
    components+=';mlir-translate;llc;opt'
    ;;
  *) exit 2 ;;
esac

# This script is hashed into the package name. Build on the named runner image.
cd "$RUNNER_TEMP"
prefix="$RUNNER_TEMP/llvm-install"
git init llvm-source
git -C llvm-source fetch --depth=1 https://github.com/ROCm/llvm-project.git "$sha"
git -C llvm-source checkout --detach FETCH_HEAD
rm -rf llvm-source/.git
cmake -G Ninja -S llvm-source/llvm -B build-llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="$targets" \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=4 \
  -DLLVM_TOOL_LTO_BUILD=OFF \
  -DLLVM_TOOL_REMARKS_SHLIB_BUILD=OFF \
  -DLLVM_DISTRIBUTION_COMPONENTS="$components" \
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
  -DCMAKE_C_VISIBILITY_PRESET=hidden \
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
cmake --build build-llvm --target install-distribution
rm -rf llvm-source build-llvm
"$prefix/bin/llvm-config" --targets-built
if [[ $variant == x86-amdgpu ]]; then
  "$prefix/bin/llvm-config" --targets-built | grep -w AMDGPU
  for tool in mlir-translate llc opt; do test -x "$prefix/bin/$tool"; done
fi
tar -C "$prefix" -I 'zstd -T0 -3' -cf "$RUNNER_TEMP/llvm.tar.zst" .
du -h "$RUNNER_TEMP/llvm.tar.zst"
