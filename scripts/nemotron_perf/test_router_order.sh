#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
SDK=$WS/build/onnx-hipdnn-ep/_therock
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
OUT=$(mktemp -d "$SOURCE/.nemotron-perf/control/router-order-XXXXXXXX")
printf 'Evidence: %s\nWaiting for shared GPU lock\n' "$OUT"
exec {lock_fd}< "$LOCK"
flock --exclusive "$lock_fd"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$OUT,target=$OUT" --workdir "$SOURCE" \
  -e "TMPDIR=$OUT" --entrypoint "$SDK/bin/hipcc" \
  hipdnn-ep-build:llvm22-restored-20260929 -O3 -std=c++17 --offload-arch=gfx1151 \
  -Ilib/Runtime/Kernels/include lib/Runtime/Kernels/hip/small_matvec_kernel.hip \
  scripts/nemotron_perf/test_router_order.hip -L"$SDK/lib" -lhipblaslt \
  -Wl,-rpath-link,"$SDK/lib" -o "$OUT/probe"
sha256sum "$SOURCE/scripts/nemotron_perf/test_router_order.hip" \
  "$SOURCE/lib/Runtime/Kernels/hip/small_matvec_kernel.hip" \
  "$OUT/probe" "$SDK/lib/libhipblaslt.so" > "$OUT/hashes.sha256"
sleep 5
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$SDK/lib" --entrypoint "$OUT/probe" \
  nemotron-main-runtime:20261020 2>&1 | tee "$OUT/run.log"
