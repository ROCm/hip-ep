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

# Create build directory structure for compiled outputs
# BUILD_DIR already points to /home/build/hip-ep-chez/lib/scheme
mkdir -p "$BUILD_DIR/libraries/mlir"
mkdir -p "$BUILD_DIR/libraries/patterns"
mkdir -p "$BUILD_DIR/libraries/passes"
mkdir -p "$BUILD_DIR/rime"

# Everything must run in ONE Scheme session for matching compilation instance IDs
$SCHEME_COMPILER <<EOF
(generate-wpo-files #t)
(compile-imported-libraries #t)

;; CRITICAL: Use pairs ("source" . "binary") to keep source tree clean
;; Source .sls files stay in source tree, compiled .so go to build tree
(library-directories
  (list (cons "$LIBRARY_DIR" "$BUILD_DIR/libraries")
        (cons "$RIME_DIR" "$BUILD_DIR/rime")))

;; compile-library creates both .so and .wpo
(compile-library "$PASS_FILE" "$BUILD_DIR/passes/${PASS_NAME}-temp.so")

;; Import to compile all dependencies with .wpo files
(import (passes $PASS_NAME))

;; Bundle everything into standalone .so
(compile-whole-library "$BUILD_DIR/passes/${PASS_NAME}-temp.wpo" 
                       "$BUILD_DIR/passes/${PASS_NAME}.so")

;; Clean up intermediate files
(for-each (lambda (f) (when (file-exists? f) (delete-file f)))
  (list "$BUILD_DIR/passes/${PASS_NAME}-temp.so"
        "$BUILD_DIR/passes/${PASS_NAME}-temp.wpo"
        "$BUILD_DIR/passes/${PASS_NAME}.wpo"))

(display "Pass compiled: ${PASS_NAME}.so\n")
EOF

SIZE=$(ls -lh "$BUILD_DIR/passes/${PASS_NAME}.so" | awk '{print $5}')
echo "Created: ${PASS_NAME}.so ($SIZE)"
