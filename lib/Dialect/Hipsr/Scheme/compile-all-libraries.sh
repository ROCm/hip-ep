#!/bin/bash
# Compile all Scheme libraries to bytecode

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

# Copy rime library
RIME_SOURCE="$SOURCE_DIR/../../../../third_party/rime/rime"
if [ ! -d "$BUILD_DIR/rime" ]; then
  echo "Copying rime library..."
  if [ -d "$RIME_SOURCE" ]; then
    cp -r "$RIME_SOURCE" "$BUILD_DIR/"
  else
    echo "Warning: rime not found at $RIME_SOURCE, skipping..."
  fi
fi

# Compile libraries using compile-library
echo "Compiling (mlir ffi)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --libdirs "$BUILD_DIR" <<EOF
(compile-library "mlir/ffi.sls" "$BUILD_DIR/mlir/ffi.so")
EOF

echo "Compiling (mlir pattern-dsl)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --libdirs "$BUILD_DIR" <<EOF
(compile-library "mlir/pattern-dsl.sls" "$BUILD_DIR/mlir/pattern-dsl.so")
EOF

echo "Compiling (mlir conversion cast)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --libdirs "$BUILD_DIR" <<EOF
(compile-library "mlir/conversion/cast.sls" "$BUILD_DIR/mlir/conversion/cast.so")
EOF

echo "Compiling (onnx-to-hipsr)..."
cd "$SOURCE_DIR/Passes"
$SCHEME_COMPILER --libdirs "$BUILD_DIR:$SOURCE_DIR/Patterns" <<EOF
(compile-library "onnx-to-hipsr.sls" "$BUILD_DIR/onnx-to-hipsr.so")
EOF

# Clean up rime .sls source files, keep only .so bytecode
echo "Cleaning up rime source files (keeping bytecode only)..."
find "$BUILD_DIR/rime" -name "*.sls" -type f -delete 2>/dev/null || true

echo "Done! All libraries compiled to bytecode."
