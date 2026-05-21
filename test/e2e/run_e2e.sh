#!/usr/bin/env bash
# E2E test runner: compile .mlir → .dll, then execute and validate.
# Args: <hip-compiler> <mlir-file> <hip-test-dll>
set -euo pipefail

HIP_COMPILER="$1"
MLIR_FILE="$2"
HIP_TEST_DLL="$3"

# Use TEST_TMPDIR (set by Bazel test runner) for isolated output.
WORK_DIR="${TEST_TMPDIR:-$(mktemp -d)}"
MODEL_NAME="$(basename "${MLIR_FILE}" .mlir)"
DLL_OUTPUT="${WORK_DIR}/${MODEL_NAME}.dll"

echo "=== Compiling: ${MLIR_FILE} ==="
"${HIP_COMPILER}" "${MLIR_FILE}" -o "${DLL_OUTPUT}"

echo "=== Executing: ${DLL_OUTPUT} ==="
cd "${WORK_DIR}"
"${HIP_TEST_DLL}" "${DLL_OUTPUT}" --verbose --validate

echo "=== PASSED: ${MODEL_NAME} ==="
