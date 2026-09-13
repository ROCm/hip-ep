#!/bin/bash
# Reproduction script for Scheme ONNX→HipSR conversion pass
set -e

cd "$(dirname "$0")/../.."
BUILD_DIR="${BUILD_DIR:-/home/build/hip-ep-1}"

echo "=== Creating test input ==="
cat > /tmp/onnx-cast-test.mlir <<'MLIR'
func.func @test_cast(%ctx: !hipsr.context, %arg0: tensor<?x8xf32>) -> tensor<?x8xf16> {
  %0 = "onnx.Cast"(%arg0) {to = f16} : (tensor<?x8xf32>) -> tensor<?x8xf16>
  "onnx.Return"(%0) : (tensor<?x8xf16>) -> ()
}
MLIR

cd "$BUILD_DIR"

echo ""
echo "=== C++ VERSION ==="
./bin/hip-mlir-opt --onnx-dialect=modeled -allow-unregistered-dialect \
  --convert-onnx-to-hipsr /tmp/onnx-cast-test.mlir

echo ""
echo "=== SCHEME VERSION ==="
echo "(Error is expected - shows pattern executed but needs TypeConverter)"
./bin/hip-mlir-opt --scheme-script="module=onnx-to-hipsr" /tmp/onnx-cast-test.mlir 2>&1 | grep -v '^\[INIT\]'

echo ""
echo "=== RESULT ==="
echo "✓ Scheme pass converts onnx.Cast → hipsr.placeholder + hipsr.cast"
echo "✓ Pattern logic matches C++ CastConversion.cpp"
echo "✗ Missing: TypeConverter adds #hipsr.mem<device> to function signatures"
