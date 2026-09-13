#!/bin/bash
# Test script to isolate FFI crash

set -e

BUILD_DIR="${BUILD_DIR:-/home/build/hip-ep-1}"
SCHEME_COMPILER="$BUILD_DIR/ChezScheme-build/ta6le/bin/ta6le/scheme"

cd "$(dirname "$0")"

echo "=== Compiling C++ test functions ==="
g++ -c -fPIC test-ffi-crash.cpp \
  -I/workspace/hip-ep/hip-ep-1/third_party/ChezScheme/boot/pb \
  -o /tmp/test-ffi-crash.o

echo "=== Creating shared library ==="
g++ -shared /tmp/test-ffi-crash.o -o /tmp/libtest-ffi-crash.so

echo "=== Compiling Scheme test ==="
$SCHEME_COMPILER --compile-imported-libraries test-ffi-crash.sls

echo "=== Running test ==="
cat > /tmp/run-test.scm <<'EOF'
(import (test-ffi-crash))

;; Load shared library
(load-shared-object "/tmp/libtest-ffi-crash.so")

;; Register functions
((foreign-procedure "register_test_functions" () void))

;; Run tests
(run-tests)
EOF

$SCHEME_COMPILER --libdirs "/tmp" --script /tmp/run-test.scm

echo "=== Test completed successfully ==="
