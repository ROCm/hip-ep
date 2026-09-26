#!/bin/bash
# Test runner script for pattern macro tests
# Sets up proper library paths for Chez Scheme

cd "$(dirname "$0")"

# Find project root (hip-ep-remote directory)
PROJECT_ROOT="$(cd ../../../.. && pwd)"

# Run tests with library paths:
# - Current directory (for (test ...) libraries)
# - libraries/ (for (mlir ...) libraries)
# - third_party/rime (for (rime loop))
exec scheme --libdirs ".:libraries:$PROJECT_ROOT/third_party/rime" \
  --script test/run-all-tests.scm
