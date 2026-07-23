#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Write hip-compiler input MLIR next to an ONNX model (<stem>.mlir).
# Stops hip-onnx-runner once mlir_bytecode_dump.mlir size is stable (skips hip-compiler).
set -euo pipefail

usage() {
  echo "Usage: onnx-model-to-onnx-mlir.sh <model.onnx> [output_dir]" >&2
  exit 2
}

[[ $# -lt 1 || $# -gt 2 ]] && usage

ONNX="$1"

WORKSPACE="${WORKSPACE:-$HOME/workspace}"
BUILD_DIR="${BUILD_DIR:-$WORKSPACE/build/hip-ep}"
CONFIG="${CONFIG:-RelWithDebInfo}"
RUNNER="$BUILD_DIR/bin/$CONFIG/hip-onnx-runner.exe"

POLL_SEC="${ONNX_TO_MLIR_POLL_SEC:-2}"
STABLE_POLLS="${ONNX_TO_MLIR_STABLE_POLLS:-3}"
TIMEOUT_SEC="${ONNX_TO_MLIR_TIMEOUT_SEC:-7200}"

if [[ ! -f "$ONNX" ]]; then
  echo "ONNX model not found: $ONNX" >&2
  exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "hip-onnx-runner not found: $RUNNER" >&2
  echo "Build: cmake --build $BUILD_DIR --config $CONFIG --target hip-onnx-runner" >&2
  exit 1
fi

stem="$(basename "$ONNX" .onnx)"
if [[ -n "${2:-}" ]]; then
  OUT_DIR="$2"
else
  OUT_DIR="$(cd "$(dirname "$ONNX")" && pwd)"
fi

mkdir -p "$OUT_DIR"

export MORPHIZEN_EP_ENABLE_CPU_DEVICE="${MORPHIZEN_EP_ENABLE_CPU_DEVICE:-1}"
export XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD="${XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD:-1073741824}"

FINAL_MLIR="$OUT_DIR/${stem}.mlir"
RAW_MLIR="$OUT_DIR/mlir_bytecode_dump.mlir"

echo "ONNX:   $ONNX"
echo "Output: $FINAL_MLIR"
echo "Runner: $RUNNER"
echo "Stop after stable: $RAW_MLIR (poll ${POLL_SEC}s, ${STABLE_POLLS} stable checks, timeout ${TIMEOUT_SEC}s)"
echo ""

stop_runner() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  kill "$pid" 2>/dev/null || true
  local waited=0
  while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt 120 ]]; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

set +e
"$RUNNER" -m "$ONNX" --dump-compiler-mlir --mlir-dump-dir "$OUT_DIR" &
runner_pid=$!
set -e

deadline=$((SECONDS + TIMEOUT_SEC))
last_size=-1
stable_count=0
dump_ready=0
stopped_early=0

while true; do
  if (( SECONDS > deadline )); then
    stop_runner "$runner_pid"
    echo "Timeout waiting for stable dump: $RAW_MLIR" >&2
    exit 1
  fi

  if [[ -f "$RAW_MLIR" ]]; then
    size="$(wc -c < "$RAW_MLIR" | tr -d ' ')"
    if [[ "$size" -gt 0 ]]; then
      if [[ "$size" == "$last_size" ]]; then
        stable_count=$((stable_count + 1))
        if [[ "$stable_count" -ge "$STABLE_POLLS" ]]; then
          dump_ready=1
          if kill -0 "$runner_pid" 2>/dev/null; then
            stopped_early=1
            stop_runner "$runner_pid"
          fi
          break
        fi
      else
        stable_count=0
        last_size="$size"
      fi
    fi
  fi

  if ! kill -0 "$runner_pid" 2>/dev/null; then
    wait "$runner_pid"
    runner_exit=$?
    if [[ -f "$RAW_MLIR" ]] && [[ "$(wc -c < "$RAW_MLIR" | tr -d ' ')" -gt 0 ]]; then
      dump_ready=1
    fi
    break
  fi

  sleep "$POLL_SEC"
done

if [[ "$dump_ready" -ne 1 ]]; then
  stop_runner "$runner_pid"
  echo "Dump not found or empty: $RAW_MLIR (runner exit ${runner_exit:-unknown})" >&2
  exit 1
fi

mv -f "$RAW_MLIR" "$FINAL_MLIR"

size_bytes=""
if command -v wc >/dev/null 2>&1; then
  size_bytes="$(wc -c < "$FINAL_MLIR" | tr -d ' ')"
fi

echo ""
echo "Output MLIR: $FINAL_MLIR"
[[ -n "$size_bytes" ]] && echo "Size: $size_bytes bytes"
if [[ "$stopped_early" -eq 1 ]]; then
  echo "(stopped hip-onnx-runner after stable pre-compiler dump; skipped post-dump hip-compiler)"
elif [[ "${runner_exit:-0}" -ne 0 ]]; then
  echo "(hip-onnx-runner exited ${runner_exit}; MLIR dump succeeded)"
fi

exit 0
