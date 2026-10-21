#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
# Generate and verify a library-specific comparison control. Do not install it.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WS=/scratch/fhanuman/RCOmEP
SDK=$WS/build/onnx-hipdnn-ep/_therock
ZSTD=$WS/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7
LOCK=$WS/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
OUT=$(mktemp -d "$SOURCE/.nemotron-perf/control/router-tuned-XXXXXXXX")
printf 'Evidence: %s\n' "$OUT"
sha256sum "$SDK/lib/libhipblaslt.so" "$SDK/bin/hipblaslt-bench" | tee "$OUT/libraries.sha256"
exec {gpu_lock}< "$LOCK"
flock --exclusive "$gpu_lock"
sleep 5
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --mount "type=bind,source=$WS,target=$WS,readonly" \
  --mount "type=bind,source=$OUT,target=$OUT" \
  --mount "type=bind,source=$ZSTD,target=$SDK/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7,readonly" \
  -e "LD_LIBRARY_PATH=$SDK/lib" --entrypoint bash nemotron-main-runtime:20261020 \
  -c 'set -euo pipefail
  sdk=$1; out=$2
  export HIPBLASLT_TUNING_FILE=$out/router.csv
  for n in 1 53 512; do
    "$sdk/bin/hipblaslt-bench" --precision f32_r -m 128 -n "$n" -k 2688 \
      --algo_method heuristic --requested_solution 60 \
      --rotating 64 --cold_iters 10 --iters 20 --use_gpu_timer --print_kernel_info
  done > "$out/tune.log" 2>&1
  unset HIPBLASLT_TUNING_FILE
  export HIPBLASLT_TUNING_OVERRIDE_FILE=$out/router.csv
  export HIPBLASLT_LOG_MASK=32
  for n in 1 53 512; do
    "$sdk/bin/hipblaslt-bench" --precision f32_r -m 128 -n "$n" -k 2688 \
      --algo_method heuristic --requested_solution 1 --cold_iters 0 --iters 1
  done > "$out/verify.log" 2>&1
  ' bash "$SDK" "$OUT"
python3 - "$OUT" <<'PY'
from pathlib import Path
import csv, re, sys
root = Path(sys.argv[1])
lines = iter((root / 'router.csv').read_text().splitlines())
expected = {}
for line in lines:
    if line.lstrip().startswith('transA,'):
        header = next(csv.reader([line.strip()]))
        values = next(csv.reader([next(lines).strip()]))
        fields = dict(zip(header, values))
        expected[fields['n']] = fields['solution_index']
assert set(expected) == {'1', '53', '512'}, expected
observed = set()
for line in (root / 'verify.log').read_text().splitlines():
    if not line.startswith('hipblaslt-bench '):
        continue
    fields = dict(re.findall(r'--?(\w+)\s+([^ ]+)', line))
    assert fields['solution_index'] == expected[fields['n']], line
    observed.add(fields['n'])
assert observed == {'1', '53', '512'}, observed
print('PASS: supported override selects the tuned solutions:', expected)
print('Tuning file:', root / 'router.csv')
PY
