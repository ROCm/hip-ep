#!/bin/bash
# Compile all Scheme libraries to bytecode
# Strategy: Compile ffi separately (uses chezscheme), then all others together
# to avoid rime compilation instance conflicts

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
mkdir -p "$BUILD_DIR/patterns"
mkdir -p "$BUILD_DIR/passes"

LIBRARY_DIR="$SOURCE_DIR/libraries"

echo "Step 1: Compile ffi + cast + passes in ONE session (all use chezscheme, no rime)..."
$SCHEME_COMPILER --libdirs "$BUILD_DIR:$SOURCE_DIR/../../../../third_party/rime" <<EOF
(library-directories (list "$LIBRARY_DIR" "$BUILD_DIR"))
(compile-library "$LIBRARY_DIR/mlir/ffi.sls" "$BUILD_DIR/mlir/ffi.so")
(compile-library "$LIBRARY_DIR/patterns/cast.sls" "$BUILD_DIR/patterns/cast.so")
(compile-library "$LIBRARY_DIR/passes/onnx-to-hipsr.sls" "$BUILD_DIR/passes/onnx-to-hipsr.so")
EOF

echo "Step 2: Compile pattern-dsl (uses rime, separate session to avoid conflicts)..."
$SCHEME_COMPILER --libdirs "$BUILD_DIR:$SOURCE_DIR/../../../../third_party/rime" <<EOF
(library-directories (list "$LIBRARY_DIR" "$BUILD_DIR"))
(compile-library "$LIBRARY_DIR/mlir/pattern-dsl.sls" "$BUILD_DIR/mlir/pattern-dsl.so")
EOF

echo "Scheme compilation complete."
