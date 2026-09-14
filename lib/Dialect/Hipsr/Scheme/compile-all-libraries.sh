#!/bin/bash
# Compile all Scheme libraries to bytecode in ONE Scheme session
# This avoids "different compilation instance" errors for shared dependencies

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

# Compile all libraries in ONE Scheme session to share compilation instances
echo "Compiling all libraries in single Scheme session..."
cd "$SOURCE_DIR/Runtime"
BUILD_DIR_ESCAPED="${BUILD_DIR//\//\\/}"  # Escape slashes for scheme strings
$SCHEME_COMPILER --libdirs "$BUILD_DIR:$SOURCE_DIR/patterns" <<EOF
;; Compile rime libraries first (expand-time dependencies)
(for-each
  (lambda (file)
    (let ((in-path (string-append "$BUILD_DIR/rime/" file ".sls"))
          (out-path (string-append "$BUILD_DIR/rime/" file ".so")))
      (when (file-exists? in-path)
        (printf "  Compiling ~a...\n" in-path)
        (compile-library in-path out-path))))
  '("loop" "control" "match" "meta" "io" "unit-test/__define-test"))

;; Compile main libraries (in dependency order)
(compile-library "mlir/ffi.sls" "$BUILD_DIR/mlir/ffi.so")
(compile-library "mlir/pattern-dsl.sls" "$BUILD_DIR/mlir/pattern-dsl.so")
(compile-library "mlir/conversion/cast.sls" "$BUILD_DIR/mlir/conversion/cast.so")

;; onnx-to-hipsr needs patterns/ in libdirs (already set above)
(compile-library "../Passes/onnx-to-hipsr.sls" "$BUILD_DIR/onnx-to-hipsr.so")

(printf "Done! All libraries compiled.\\\\n")
EOF

echo "Scheme compilation complete."
