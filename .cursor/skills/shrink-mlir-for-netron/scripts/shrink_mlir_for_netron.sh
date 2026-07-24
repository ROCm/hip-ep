#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Shrink huge MLIR and apply Netron fixes -> <stem>.netron.mlir only (intermediate in temp)
set -euo pipefail

usage() {
  echo "Usage: shrink_mlir_for_netron.sh <input.mlir> [output_dir] [--keep-constants]" >&2
  exit 2
}

[[ $# -lt 1 ]] && usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHRINK="$SCRIPT_DIR/shrink_mlir_for_viewer.py"
NETRON="$SCRIPT_DIR/netron_fixes.py"

INPUT="$1"
shift

OUT_DIR=""
KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-constants) KEEP=1; shift ;;
    *)
      if [[ -z "$OUT_DIR" ]]; then
        OUT_DIR="$1"
        shift
      else
        usage
      fi
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
fi

stem="$(basename "$INPUT" .mlir)"
NETRON_OUT="$OUT_DIR/${stem}.netron.mlir"

VIEW="$(mktemp "${TMPDIR:-/tmp}/${stem}.view.XXXXXX.mlir")"
cleanup() { rm -f "$VIEW"; }
trap cleanup EXIT

PYTHON="$(command -v python3 >/dev/null 2>&1 && echo python3 || echo python)"
"$PYTHON" "$SHRINK" "$INPUT" -o "$VIEW" >/dev/null

NETRON_ARGS=("$NETRON" "$VIEW" -o "$NETRON_OUT")
[[ "$KEEP" -eq 1 ]] && NETRON_ARGS+=(--keep-constants)
"$PYTHON" "${NETRON_ARGS[@]}"

echo ""
echo "Netron file: $NETRON_OUT"
