#!/bin/bash
# Test script for R6RS Scheme integration

set -e

REMOTE_HOST="xcoengvm226019"
REMOTE_PORT="23762"
BUILD_DIR="/home/build/hip-ep-1"
TEST_INPUT="/tmp/test-cast.mlir"
TEST_OUTPUT="/tmp/test-cast-output.mlir"

echo "=== Step 1: Build System ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $BUILD_DIR && \
  rm -rf lib/scheme CMakeFiles lib/Dialect/Hipsr/Scheme && \
  cmake /workspace/hip-ep/hip-ep-1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja -DLIT_EXECUTABLE=/home/local/bin/lit >/dev/null 2>&1 && \
  ninja -j16 SchemeLibraries HipsrSchemePass hip-mlir-opt 2>&1 | tail -5
"

echo ""
echo "=== Step 2: Verify Bytecode Output ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "
  ls -lh $BUILD_DIR/lib/scheme/mlir/*.so && \
  ls -lh $BUILD_DIR/lib/scheme/mlir/conversion/*.so
"

echo ""
echo "=== Step 3: Create Test Input ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "cat > $TEST_INPUT" <<'EOF'
module {
  func.func @test_cast(
      %ctx: !hipsr.context,
      %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
    %0 = "onnx.Cast"(%input) {to = f16} : (tensor<?x8xf32>) -> tensor<?x8xf16>
    return %0 : tensor<?x8xf16>
  }
}
EOF

echo "Test input created at $TEST_INPUT"

echo ""
echo "=== Step 4: Run Scheme-based Pass ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $BUILD_DIR && \
  ./bin/hip-mlir-opt \
    --scheme-script='scriptName=CastConversion-r6rs.scm logLevel=info' \
    $TEST_INPUT \
    -o $TEST_OUTPUT 2>&1 | grep -E '\[info\]|\[error\]|error:' || true
"

echo ""
echo "=== Step 5: Check Output ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "cat $TEST_OUTPUT"

echo ""
echo "=== Step 6: Verify Transformation ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "
  echo 'Checking function signature has device memory space:' && \
  grep -q 'tensor<?x8xf32, #hipsr.mem<device>>' $TEST_OUTPUT && echo '  ✓ Found device memory space in input' || echo '  ✗ Missing device memory space' && \
  \
  echo 'Checking hipsr.placeholder exists:' && \
  grep -q 'hipsr.placeholder' $TEST_OUTPUT && echo '  ✓ Found hipsr.placeholder' || echo '  ✗ Missing hipsr.placeholder' && \
  \
  echo 'Checking hipsr.cast exists:' && \
  grep -q 'hipsr.cast' $TEST_OUTPUT && echo '  ✓ Found hipsr.cast' || echo '  ✗ Missing hipsr.cast' && \
  \
  echo 'Checking onnx.Cast removed:' && \
  ! grep -q 'onnx.Cast' $TEST_OUTPUT && echo '  ✓ onnx.Cast removed' || echo '  ✗ onnx.Cast still present'
"

echo ""
echo "=== Step 7: Run MLIR Verifier ==="
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $BUILD_DIR && \
  ./bin/hip-mlir-opt --verify-each $TEST_OUTPUT -o /dev/null 2>&1 && \
  echo '✓ Verification PASSED' || \
  (echo '✗ Verification FAILED:' && ./bin/hip-mlir-opt --verify-each $TEST_OUTPUT -o /dev/null)
"

echo ""
echo "=== Test Complete ==="
