#!/bin/bash
# Compile a single pass into a standalone .so file with all dependencies bundled

set -e

if [ $# -ne 4 ]; then
    echo "Usage: $0 <scheme-compiler> <source-dir> <build-dir> <pass-name>"
    echo "Example: $0 /path/to/scheme /path/to/Scheme /path/to/build onnx-to-hipsr"
    exit 1
fi

SCHEME_COMPILER=$1
SOURCE_DIR=$2
BUILD_DIR=$3
PASS_NAME=$4

echo "Compiling pass: $PASS_NAME"

mkdir -p "$BUILD_DIR/passes"

LIBRARY_DIR="$SOURCE_DIR/libraries"
RIME_DIR="$SOURCE_DIR/../../../../third_party/rime"
PASS_FILE="$LIBRARY_DIR/passes/${PASS_NAME}.sls"

if [ ! -f "$PASS_FILE" ]; then
    echo "Error: Pass file not found: $PASS_FILE"
    exit 1
fi

# Everything must run in ONE Scheme session for matching compilation instance IDs
$SCHEME_COMPILER <<EOF
(generate-wpo-files #t)
(compile-imported-libraries #t)
(library-directories (list "$LIBRARY_DIR" "$BUILD_DIR" "$RIME_DIR"))

; compile-library creates both .so and .wpo
(compile-library "$PASS_FILE" "$BUILD_DIR/passes/${PASS_NAME}-temp.so")

; Import to compile all dependencies with .wpo files
(import (passes $PASS_NAME))

; Bundle everything into standalone .so
(compile-whole-library "$BUILD_DIR/passes/${PASS_NAME}-temp.wpo" 
                       "$BUILD_DIR/passes/${PASS_NAME}.so")

; Clean up intermediate files
(for-each (lambda (f) (when (file-exists? f) (delete-file f)))
  (list "$BUILD_DIR/passes/${PASS_NAME}-temp.so"
        "$BUILD_DIR/passes/${PASS_NAME}-temp.wpo"
        "$BUILD_DIR/passes/${PASS_NAME}.wpo"))

(display "Pass compiled: ${PASS_NAME}.so\n")
EOF

SIZE=$(ls -lh "$BUILD_DIR/passes/${PASS_NAME}.so" | awk '{print $5}')
echo "Created: ${PASS_NAME}.so ($SIZE)"
