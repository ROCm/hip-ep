#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Generate ONNX test models for E2E integration tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="../../build/$(basename "$(cd "$SCRIPT_DIR/../.." && pwd)")"

echo "Generating ONNX test models..."
echo "Build directory: $BUILD_DIR"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Generate two-layer Conv model (similar to demo_two_layer_conv.mlir)
echo ""
echo "Generating two-layer Conv model..."
python3 "$SCRIPT_DIR/gen_conv_model.py" \
  --two-layer \
  --output "$BUILD_DIR/conv_model.onnx"

# Generate Conv+Gemm model
echo ""
echo "Generating Conv+Gemm model..."
python3 "$SCRIPT_DIR/gen_conv_gemm_model.py" \
  --output "$BUILD_DIR/conv_gemm_model.onnx"

echo ""
echo "Models generated successfully in: $BUILD_DIR"
echo "  - conv_model.onnx (two-layer Conv)"
echo "  - conv_gemm_model.onnx (Conv + Gemm)"
echo ""
echo "Run integration test with:"
echo "  ctest --test-dir $BUILD_DIR -R OrtIntegration --verbose"
