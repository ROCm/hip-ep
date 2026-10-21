#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
OUT=$(mktemp -d "$SOURCE/.nemotron-perf/control/la-direct-XXXXXXXX")
SDK=$WS/build/onnx-hipdnn-ep/_therock
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
LOG=$OUT/run.log
REFERENCE=$SOURCE/.nemotron-perf/install-sparse-v18/lib
printf 'Evidence: %s\nWaiting for shared GPU lock: %s\n' "$OUT" "$LOCK"
exec {lock_fd}< "$LOCK"
flock --exclusive "$lock_fd"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$OUT,target=$OUT" --workdir "$SOURCE" \
  -e "TMPDIR=$OUT" --entrypoint "$SDK/bin/hipcc" \
  hipdnn-ep-build:llvm22-restored-20260929 -O3 -std=c++17 \
  --offload-arch=gfx1151 -Ilib/Runtime/Kernels/include \
  lib/Runtime/Kernels/hip/linear_attention_kernel.hip \
  scripts/nemotron_perf/test_linear_attention.cpp \
  -L"$REFERENCE" -lcustom_kernels_gfx1151 -ldl -Wl,-rpath-link,"$SDK/lib" \
  -o "$OUT/test_linear_attention"
printf 'GPU lock acquired; cooldown: 5 seconds\n'
sleep 5
sha256sum "$SOURCE/lib/Runtime/Kernels/hip/linear_attention_kernel.hip" \
  "$SOURCE/scripts/nemotron_perf/test_linear_attention.cpp" \
  "$OUT/test_linear_attention" | tee "$LOG"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$REFERENCE:$SDK/lib" \
  -e "NEMOTRON_LA_REFERENCE_LIB=$REFERENCE/libcustom_kernels_gfx1151.so" \
  -e "HIPDNN_EP_EXPERIMENTAL_LA_FIXED_GEOMETRY=${NEMOTRON_LA_FIXED_GEOMETRY:-0}" \
  --entrypoint "$OUT/test_linear_attention" \
  nemotron-main-runtime:20261020 2>&1 | tee -a "$LOG"
