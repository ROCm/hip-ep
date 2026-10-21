#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
OUT=$SOURCE/.nemotron-perf
RUN=$(mktemp -d "$OUT/control/routed-XXXXXXXX")
SDK=$WS/build/onnx-hipdnn-ep/_therock
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
LOG=$RUN/run.log
REFERENCE=$OUT/install-sparse-v5/lib
SPARSE_REFERENCE=${NEMOTRON_SPARSE_REFERENCE_LIB:-$OUT/install-sparse-v13/lib/libcustom_kernels_gfx1151.so}
EXTRA=()
TILE=${NEMOTRON_ROUTED_TILE:-2}
case "$TILE" in 1|2|4|8) ;; *) printf 'Invalid test tile: %s\n' "$TILE" >&2; exit 1;; esac
EXE=$RUN/test_routed_experts
if [[ ${NEMOTRON_ROUTED_COLD_ONLY:-0} == 1 ]]; then
  EXE=$RUN/test_routed_experts_tile${TILE}_cold
fi
if [[ ${NEMOTRON_DENSE_REFERENCE:-0} == 1 ]]; then
  EXTRA=(-DNEMOTRON_DENSE_REFERENCE -L"$REFERENCE" -lcustom_kernels_gfx1151
         -Wl,-rpath-link,"$SDK/lib")
fi

# CPU compilation and GPU measurement share this APU's power budget.
# Hold the lease for both, then cool down before the GPU test.
printf 'Waiting for shared GPU lock: %s\n' "$LOCK"
exec {lock_fd}< "$LOCK"
flock --exclusive "$lock_fd"
printf 'GPU lock acquired; building direct test\n'
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$RUN,target=$RUN" --workdir "$SOURCE" \
  -e "TMPDIR=$RUN" --entrypoint "$SDK/bin/hipcc" \
  hipdnn-ep-build:llvm22-restored-20260929 -O3 -std=c++17 \
  --offload-arch=gfx1151 -Ilib/Runtime/Kernels/include \
  "-DNEMOTRON_ROUTED_W4A8_TILE=$TILE" \
  lib/Runtime/Kernels/hip/routed_experts_kernel.hip \
  scripts/nemotron_perf/test_routed_experts.cpp "${EXTRA[@]}" -ldl -o "$EXE"

printf 'Build complete; cooldown: 5 seconds\n'
sleep 5
sha256sum "$SOURCE/lib/Runtime/Kernels/hip/routed_experts_kernel.hip" \
  "$SOURCE/scripts/nemotron_perf/test_routed_experts.cpp" \
  "$EXE" | tee "$LOG"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$REFERENCE:$SDK/lib" \
  -e "NEMOTRON_SPARSE_REFERENCE_LIB=$SPARSE_REFERENCE" \
  -e "NEMOTRON_ROUTED_COLD_ONLY=${NEMOTRON_ROUTED_COLD_ONLY:-0}" \
  --entrypoint "$EXE" \
  nemotron-main-runtime:20261020 2>&1 | tee -a "$LOG"
