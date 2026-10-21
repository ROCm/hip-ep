#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
OUT=$SOURCE/.nemotron-perf
RUN=$(mktemp -d "$OUT/control/swiglu-XXXXXXXX")
SDK=$WS/build/onnx-hipdnn-ep/_therock
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
REFERENCE=$OUT/install-sparse-v12/lib
KERNEL=lib/Runtime/Kernels/hip/swiglu_f32_kernel.hip
EXE=$RUN/test_swiglu_f32
DEFINES=(-DNEMOTRON_SWIGLU_BCAST)
LOG=$RUN/run.log
exec {lock_fd}< "$LOCK"
printf 'Evidence: %s\nWaiting for shared GPU lock\n' "$RUN"
flock --exclusive "$lock_fd"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$RUN,target=$RUN" --workdir "$SOURCE" \
  -e "TMPDIR=$RUN" --entrypoint "$SDK/bin/hipcc" \
  hipdnn-ep-build:llvm22-restored-20260929 -O3 -std=c++17 \
  --offload-arch=gfx1151 -Ilib/Runtime/Kernels/include \
  "${DEFINES[@]}" "$KERNEL" scripts/nemotron_perf/test_swiglu_f32.cpp \
  -L"$REFERENCE" -lcustom_kernels_gfx1151 -Wl,-rpath-link,"$SDK/lib" \
  -o "$EXE"
printf 'GPU lock acquired; cooldown: 5 seconds\n'
sleep 5
sha256sum "$KERNEL" \
  scripts/nemotron_perf/test_swiglu_f32.cpp "$EXE" \
  "$REFERENCE/libcustom_kernels_gfx1151.so" | tee "$LOG"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$REFERENCE:$SDK/lib" --entrypoint "$EXE" \
  nemotron-main-runtime:20261020 2>&1 | tee -a "$LOG"
