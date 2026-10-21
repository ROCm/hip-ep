#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
ROOT=$SOURCE/.nemotron-perf
OUT=$(mktemp -d "$ROOT/control/top-k-XXXXXXXX")
SDK=$WS/build/onnx-hipdnn-ep/_therock
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
LOG=$OUT/run.log
exec {lock_fd}< "$LOCK"
printf 'Evidence: %s\nWaiting for shared GPU lock\n' "$OUT"
flock --exclusive "$lock_fd"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$OUT,target=$OUT" --workdir "$SOURCE" \
  -e "TMPDIR=$OUT" --entrypoint bash hipdnn-ep-build:llvm22-restored-20260929 -c '
set -euo pipefail
"$1/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx1151 -Ilib/Runtime/Kernels/include \
    -Dhip_top_k=hip_top_k_baseline -c "$3" -o "$2/top_k_baseline.o"
"$1/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx1151 -Ilib/Runtime/Kernels/include \
    lib/Runtime/Kernels/hip/top_k_kernel.hip scripts/nemotron_perf/test_top_k.cpp \
    -x none "$2/top_k_baseline.o" -o "$2/test_top_k"
' bash "$SDK" "$OUT" "$ROOT/top_k_kernel_baseline.hip"
printf 'GPU lock acquired; cooldown: 5 seconds\n'
sleep 5
sha256sum lib/Runtime/Kernels/hip/top_k_kernel.hip scripts/nemotron_perf/test_top_k.cpp \
  "$ROOT/top_k_kernel_baseline.hip" "$OUT/test_top_k" | tee "$LOG"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$SDK/lib" --entrypoint "$OUT/test_top_k" \
  nemotron-main-runtime:20261020 2>&1 | tee -a "$LOG"
