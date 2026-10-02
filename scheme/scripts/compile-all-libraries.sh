#!/bin/bash
# Compile all top-level passes into individual standalone .so files

set -e

SCHEME_COMPILER=$1
SOURCE_DIR=$2
BUILD_DIR=$3

echo "Compiling all passes..."
echo "Compiler: $SCHEME_COMPILER"
echo "Source: $SOURCE_DIR"
echo "Output: $BUILD_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILE_PASS="$SCRIPT_DIR/compile-pass.sh"

# List of all top-level passes to compile
PASSES=(
    "onnx-to-hipsr"
    "print"
)

# Compile each pass independently
for PASS in "${PASSES[@]}"; do
    echo ""
    echo "=== Compiling $PASS ==="
    "$COMPILE_PASS" "$SCHEME_COMPILER" "$SOURCE_DIR" "$BUILD_DIR" "$PASS"
done

echo ""
echo "=================================="
echo "All passes compiled:"
find "$BUILD_DIR" -name '*.so' -type f | while read f; do
    size=$(ls -lh "$f" | awk '{print $5}')
    name=$(basename "$f")
    printf "  %-30s %s\n" "$name" "$size"
done | sort

echo ""
echo "Total: $(find "$BUILD_DIR" -name '*.so' -exec ls -l {} \; | awk '{sum+=$5} END {printf "%.1f KB", sum/1024}')"
