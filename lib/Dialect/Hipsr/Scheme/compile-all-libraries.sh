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

# Copy rime library if not already there
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

# Library search path (BUILD_DIR needs to be in path for rime)
LIBDIRS="$SOURCE_DIR/Runtime:$SOURCE_DIR/Passes:$BUILD_DIR"

# Compile each library
echo "Compiling (mlir ffi)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir ffi))")
mv mlir/ffi.so "$BUILD_DIR/mlir/"

echo "Compiling (mlir pattern-dsl)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir pattern-dsl))")
mv mlir/pattern-dsl.so "$BUILD_DIR/mlir/"

echo "Compiling (mlir conversion cast)..."
cd "$SOURCE_DIR/Runtime"
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (mlir conversion cast))")
mv mlir/conversion/cast.so "$BUILD_DIR/mlir/conversion/"

echo "Compiling (onnx-to-hipsr)..."
cd "$SOURCE_DIR/Passes"
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIBDIRS" --program <(echo "(import (onnx-to-hipsr))")
mv onnx-to-hipsr.so "$BUILD_DIR/"

# Clean up any .so files in source
find "$SOURCE_DIR" -name "*.so" -type f -delete

echo "Done! All libraries compiled to bytecode."
