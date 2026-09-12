#!/bin/bash
# Compile all Scheme libraries to bytecode
# Super simple - just compile everything from scratch

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
# Note: SOURCE_DIR is lib/Dialect/Hipsr/Scheme
# We need to go up to workspace root: ../../../../
RIME_SOURCE="$SOURCE_DIR/../../../../third_party/rime/rime"
if [ ! -d "$BUILD_DIR/rime" ]; then
  echo "Copying rime library..."
  if [ -d "$RIME_SOURCE" ]; then
    cp -r "$RIME_SOURCE" "$BUILD_DIR/"
  else
    echo "Warning: rime not found at $RIME_SOURCE, skipping..."
  fi
fi

# Library search path
# Important: BUILD_DIR first so compiled .so files are found before recompiling
LIBDIRS="$BUILD_DIR:$SOURCE_DIR/Runtime:$SOURCE_DIR/Passes"

# Compile libraries in dependency order
# Once a library is compiled, later compilations find it in BUILD_DIR

echo "Compiling (mlir ffi)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir ffi))")
mv mlir/ffi.so "$BUILD_DIR/mlir/"

echo "Compiling (mlir pattern-dsl)..."
cd "$SOURCE_DIR/Runtime"
# This will find (mlir ffi) in BUILD_DIR, not recompile it
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir pattern-dsl))")
mv mlir/pattern-dsl.so "$BUILD_DIR/mlir/"

echo "Compiling (mlir conversion cast)..."
cd "$SOURCE_DIR/Runtime"
# This will find (mlir ffi) and (mlir pattern-dsl) in BUILD_DIR
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir conversion cast))")
mv mlir/conversion/cast.so "$BUILD_DIR/mlir/conversion/"

echo "Compiling (onnx-to-hipsr)..."
cd "$SOURCE_DIR/Passes"
# This will find all dependencies in BUILD_DIR
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (onnx-to-hipsr))")
mv onnx-to-hipsr.so "$BUILD_DIR/"

# Clean up any .so files in source tree
find "$SOURCE_DIR" -name "*.so" -type f -delete

# Clean up rime .sls source files, keep only .so bytecode
# This avoids "different compilation instance" errors
echo "Cleaning up rime source files (keeping bytecode only)..."
find "$BUILD_DIR/rime" -name "*.sls" -type f -delete 2>/dev/null || true

echo "Done! All libraries compiled to bytecode."
