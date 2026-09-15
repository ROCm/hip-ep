#!/bin/bash
# Compile all Scheme libraries to bytecode
# Each library compiled in SEPARATE process to work around ChezScheme 10.4.1 bug
# where libraries using (chezscheme) break subsequent compilations in same session

set -e

SCHEME_COMPILER=$1
SOURCE_DIR=$2
BUILD_DIR=$3

echo "Compiling all Scheme libraries..."
echo "Compiler: $SCHEME_COMPILER"
echo "Source: $SOURCE_DIR"
echo "Output: $BUILD_DIR"

# Create output directories
mkdir -p "$BUILD_DIR/mlir"
mkdir -p "$BUILD_DIR/mlir/conversion"

# Copy rime library (always refresh to avoid version mismatches)
RIME_SOURCE="$SOURCE_DIR/../../../../third_party/rime/rime"
echo "Refreshing rime library..."
rm -rf "$BUILD_DIR/rime"
if [ -d "$RIME_SOURCE" ]; then
  cp -r "$RIME_SOURCE" "$BUILD_DIR/"
else
  echo "Warning: rime not found at $RIME_SOURCE, skipping..."
fi

# Compile rime libraries first (each in separate process)
echo "Compiling rime libraries..."
for file in loop unit-test/__define-test; do
  IN_PATH="$BUILD_DIR/rime/$file.sls"
  OUT_PATH="$BUILD_DIR/rime/$file.so"
  if [ -f "$IN_PATH" ]; then
    echo "  $file..."
    $SCHEME_COMPILER --libdirs "$BUILD_DIR" <<EOF
(compile-library "$IN_PATH" "$OUT_PATH")
EOF
  fi
done

# Compile main libraries
# Strategy: mlir/ffi in separate session (it uses chezscheme and corrupts environment)
# All others in ONE session (to share rime compilation instances)
cd "$SOURCE_DIR/Runtime"

echo "Compiling mlir/ffi (separate session due to ChezScheme bug)..."
$SCHEME_COMPILER <<EOF
(compile-library "mlir/ffi.sls" "$BUILD_DIR/mlir/ffi.so")
EOF

echo "Compiling remaining libraries (single session to share rime instances)..."
$SCHEME_COMPILER --libdirs "$BUILD_DIR:$SOURCE_DIR" <<EOF
;; Compile all libraries that depend on rime in ONE session
;; This ensures they share the same compilation instance of rime
(compile-library "mlir/pattern-dsl.sls" "$BUILD_DIR/mlir/pattern-dsl.so")
(compile-library "mlir/conversion/cast.sls" "$BUILD_DIR/mlir/conversion/cast.so")
(compile-library "../Passes/onnx-to-hipsr.sls" "$BUILD_DIR/onnx-to-hipsr.so")
(display "All libraries compiled successfully!\n")
EOF

echo "Scheme compilation complete."
