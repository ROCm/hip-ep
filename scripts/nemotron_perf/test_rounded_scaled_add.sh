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
REFERENCE=$SOURCE/.nemotron-perf/install-checkpoint-core-v1/lib
OUT=$(mktemp -d "$SOURCE/.nemotron-perf/control/rounded-add-XXXXXXXX")
printf 'Evidence: %s\nWaiting for shared GPU lock\n' "$OUT"
exec {lock_fd}< "$LOCK"
flock --exclusive "$lock_fd"
docker run --rm --network none --user "$(id -u):$(id -g)" \
 --mount "type=bind,source=$WS,target=$WS,readonly" \
 --mount "type=bind,source=$OUT,target=$OUT" --workdir "$SOURCE" \
 -e "TMPDIR=$OUT" --entrypoint "$SDK/bin/hipcc" \
 hipdnn-ep-build:llvm22-restored-20260929 -O3 -std=c++17 --offload-arch=gfx1151 \
 -Ilib/Runtime/Kernels/include lib/Runtime/Kernels/hip/rounded_scaled_add_kernel.hip \
 scripts/nemotron_perf/test_rounded_scaled_add.cpp -L"$REFERENCE" -lcustom_kernels_gfx1151 \
 -Wl,-rpath-link,"$SDK/lib" -o "$OUT/test"
sleep 5
sha256sum lib/Runtime/Kernels/hip/rounded_scaled_add_kernel.hip \
 scripts/nemotron_perf/test_rounded_scaled_add.cpp "$OUT/test" \
 "$REFERENCE/libcustom_kernels_gfx1151.so" > "$OUT/hashes.sha256"
docker run --rm --network none --user "$(id -u):$(id -g)" \
 --device /dev/kfd --device /dev/dri/renderD128 \
 --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
 --mount "type=bind,source=$WS,target=$WS,readonly" \
 --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
 -e "LD_LIBRARY_PATH=$REFERENCE:$SDK/lib" --entrypoint "$OUT/test" \
 nemotron-main-runtime:20261020 2>&1 | tee "$OUT/run.log"
