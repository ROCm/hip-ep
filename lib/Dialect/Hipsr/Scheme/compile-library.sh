#!/bin/bash
# Compile a Scheme library by creating a temporary program that imports it
# Outputs .so bytecode to the build directory, not the source directory

set -e

SCHEME_COMPILER=$1
LIBRARY_NAME=$2  # Space-separated, e.g. "mlir ffi"
LIB_DIRS=$3
SOURCE_DIR=$4
BUILD_DIR=$5
OUTPUT_FILE=$6

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Create temporary program
TEMP_PROG=$(mktemp /tmp/compile-XXXXXX.scm)
echo "(import ($LIBRARY_NAME))" > $TEMP_PROG

# Chez Scheme always outputs .so next to the .sls source file
# Solution: compile in source, then move ALL .so to build directory and clean source

# Compile (outputs to source directory)
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIB_DIRS" --program $TEMP_PROG

# Move compiled .so from source to build output location
LIBRARY_PATH=$(echo "$LIBRARY_NAME" | tr ' ' '/')
mv "${SOURCE_DIR}/${LIBRARY_PATH}.so" "$OUTPUT_FILE"

# Clean up ALL .so files from source directory to prevent pollution
find "$SOURCE_DIR" -name "*.so" -type f -delete

# Clean up temp file
rm -f $TEMP_PROG
