#!/bin/bash
# Compile a Scheme library by creating a temporary program that imports it

SCHEME_COMPILER=$1
LIBRARY_NAME=$2  # Space-separated, e.g. "mlir ffi"
LIB_DIRS=$3
SOURCE_DIR=$4
OUTPUT_FILE=$5

# Create temporary program
TEMP_PROG=$(mktemp /tmp/compile-XXXXXX.scm)
echo "(import ($LIBRARY_NAME))" > $TEMP_PROG

# Compile
$SCHEME_COMPILER --compile-imported-libraries --libdirs "$LIB_DIRS" --program $TEMP_PROG

# Clean up
rm -f $TEMP_PROG

# Move compiled .so to output location
# Convert "mlir ffi" to "mlir/ffi.so"
LIBRARY_PATH=$(echo "$LIBRARY_NAME" | tr ' ' '/')
mv "${SOURCE_DIR}/${LIBRARY_PATH}.so" "$OUTPUT_FILE"
