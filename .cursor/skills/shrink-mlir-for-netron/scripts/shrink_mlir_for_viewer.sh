#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Shrink huge MLIR for Netron / diff (streaming Python tool).
set -euo pipefail

usage() {
  echo "Usage: shrink_mlir_for_viewer.sh <input.mlir> [-o output.mlir]" >&2
  exit 2
}

[[ $# -lt 1 ]] && usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/shrink_mlir_for_viewer.py"

INPUT="$1"
shift
exec python "$PY" "$INPUT" "$@"
