#!/bin/bash
# Compile top-level passes as standalone libraries with all dependencies bundled

set -e

SCHEME_COMPILER=$1
SOURCE_DIR=$2
BUILD_DIR=$3

echo "Compiling standalone passes (all dependencies bundled)..."
echo "Compiler: $SCHEME_COMPILER"
echo "Source: $SOURCE_DIR"
echo "Output: $BUILD_DIR"

mkdir -p "$BUILD_DIR/passes"

LIBRARY_DIR="$SOURCE_DIR/libraries"
RIME_DIR="$SOURCE_DIR/../../../../third_party/rime"

# CRITICAL: Everything must run in ONE Scheme session
$SCHEME_COMPILER <<EOF
(generate-wpo-files #t)
(compile-imported-libraries #t)
(library-directories (list "$LIBRARY_DIR" "$BUILD_DIR" "$RIME_DIR"))

; Step 1: Compile top-level passes to generate WPO files
(compile-library "$LIBRARY_DIR/passes/onnx-to-hipsr.sls" "$BUILD_DIR/passes/onnx-to-hipsr-temp.so")
(compile-library "$LIBRARY_DIR/passes/print.sls" "$BUILD_DIR/passes/print-temp.so")

; Step 2: Import to compile all dependencies (with WPO files)
(import (passes onnx-to-hipsr))
(import (passes print))

; Step 3: Bundle everything into standalone .so files
(compile-whole-library "$BUILD_DIR/passes/onnx-to-hipsr-temp.wpo" "$BUILD_DIR/passes/onnx-to-hipsr.so")
(compile-whole-library "$BUILD_DIR/passes/print-temp.wpo" "$BUILD_DIR/passes/print.so")

; Step 4: Clean up intermediate files
(for-each (lambda (f) (when (file-exists? f) (delete-file f)))
  (list "$BUILD_DIR/passes/onnx-to-hipsr-temp.so"
        "$BUILD_DIR/passes/onnx-to-hipsr-temp.wpo"
        "$BUILD_DIR/passes/print-temp.so"
        "$BUILD_DIR/passes/print-temp.wpo"
        "$BUILD_DIR/passes/onnx-to-hipsr.wpo"
        "$BUILD_DIR/passes/print.wpo"))

(display "Standalone passes created\n")
EOF

echo ""
echo "Compilation complete:"
find "$BUILD_DIR" -name '*.so' -type f | while read f; do
    size=$(ls -lh "$f" | awk '{print $5}')
    name=$(basename "$f")
    printf "  %-30s %s\n" "$name" "$size"
done | sort

echo ""
echo "Total: $(find "$BUILD_DIR" -name '*.so' -exec ls -l {} \; | awk '{sum+=$5} END {printf "%.1f KB", sum/1024}')"
echo "All dependencies bundled. No source files needed."
