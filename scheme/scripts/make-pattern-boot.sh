#!/bin/bash
# Create a custom boot file containing all pattern DSL libraries

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <scheme-compiler> <source-dir> <output-boot-file>"
    echo "Example: $0 /path/to/scheme /path/to/Scheme patterns.boot"
    exit 1
fi

SCHEME=$1
SOURCE_DIR=$2
OUTPUT_BOOT=$3

echo "Creating pattern DSL boot file: $OUTPUT_BOOT"
echo "Scheme compiler: $SCHEME"
echo "Source directory: $SOURCE_DIR"

LIBRARY_DIR="$SOURCE_DIR/libraries"
RIME_DIR="$SOURCE_DIR/../../../../third_party/rime"
BUILD_TEMP="$SOURCE_DIR/boot-temp"

mkdir -p "$BUILD_TEMP"

# Clean any existing .so files to force fresh compilation
echo "Cleaning old .so files..."
find "$LIBRARY_DIR" -name "*.so" -delete 2>/dev/null || true
find "$LIBRARY_DIR" -name "*.wpo" -delete 2>/dev/null || true

# Compile all pattern libraries in dependency order
echo "Compiling pattern libraries..."
$SCHEME --libdirs "$LIBRARY_DIR:$RIME_DIR" <<'EOF'
(import (chezscheme))

;; Compile each library to .so (in dependency order)
(define libs
  '("mlir/pattern-keywords"
    "mlir/pattern-ast"
    "mlir/pattern-parse"
    "mlir/pattern-validate"
    "mlir/pattern-actions"
    "mlir/pattern-analyze"
    "mlir/pattern-codegen"
    "mlir/pattern-macro"
    "mlir/ffi"
    "patterns/cast"))

(for-each
  (lambda (lib)
    (let ([path (string-append "libraries/" lib ".sls")])
      (printf "Compiling ~a...~n" lib)
      (compile-file path)))
  libs)

(printf "All libraries compiled.~n")
EOF

# Collect all .so files
echo "Collecting compiled libraries..."
SO_FILES=""
for lib in \
  "mlir/pattern-keywords" \
  "mlir/pattern-ast" \
  "mlir/pattern-parse" \
  "mlir/pattern-validate" \
  "mlir/pattern-actions" \
  "mlir/pattern-analyze" \
  "mlir/pattern-codegen" \
  "mlir/pattern-macro" \
  "mlir/ffi" \
  "patterns/cast"
do
  SO_FILE="$LIBRARY_DIR/${lib}.so"
  if [ ! -f "$SO_FILE" ]; then
    echo "Error: Expected .so file not found: $SO_FILE"
    exit 1
  fi
  SO_FILES="$SO_FILES \"$SO_FILE\""
done

# Create boot file using $make-boot-file
echo "Creating boot file..."
$SCHEME <<'EOF'
(import (chezscheme))

;; Create boot file with all pattern libraries
;; Depends on both petite and scheme boots
(make-boot-file "patterns.boot"
                '("petite" "scheme")
                "./libraries/mlir/pattern-keywords.so"
                "./libraries/mlir/pattern-ast.so"
                "./libraries/mlir/pattern-parse.so"
                "./libraries/mlir/pattern-validate.so"
                "./libraries/mlir/pattern-actions.so"
                "./libraries/mlir/pattern-analyze.so"
                "./libraries/mlir/pattern-codegen.so"
                "./libraries/mlir/pattern-macro.so"
                "./libraries/mlir/ffi.so"
                "./libraries/patterns/cast.so")

(printf "Boot file created: patterns.boot~n")
EOF

echo ""
echo "Success! Boot file created: $OUTPUT_BOOT"
ls -lh "$OUTPUT_BOOT"

# Test loading the boot file
echo ""
echo "Testing boot file..."
$SCHEME -b petite.boot -b scheme.boot -b "$OUTPUT_BOOT" --script - <<'TESTEOF'
(import (chezscheme))

;; Try to use the pattern macro
(printf "Testing if pattern libraries loaded...~n")

;; Check if pattern-macro exports are available
(guard (ex [else (printf "Error: ~a~n" ex) (exit 1)])
  (eval '(import (mlir pattern-macro)))
  (printf "Successfully imported (mlir pattern-macro)~n")

  (eval '(import (mlir ffi)))
  (printf "Successfully imported (mlir ffi)~n")

  (eval '(import (patterns cast)))
  (printf "Successfully imported (patterns cast)~n"))

(printf "~nBoot file test PASSED!~n")
TESTEOF

echo ""
echo "Boot file is ready for deployment!"
