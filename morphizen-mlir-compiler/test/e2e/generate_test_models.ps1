##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Generate ONNX test models for E2E integration tests

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Get the actual project root (onnx-hipdnn-ep-mlir-integration-1)
# Script is at mlir-compiler/test/e2e, so go up 3 levels
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$ProjectName = Split-Path -Leaf $ProjectRoot
$BuildDir = Join-Path $ProjectRoot "build\$ProjectName"

Write-Host "Generating ONNX test models..."
Write-Host "Build directory: $BuildDir"

# Create build directory if it doesn't exist
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Generate two-layer Conv model (similar to demo_two_layer_conv.mlir)
Write-Host ""
Write-Host "Generating two-layer Conv model..."
python "$ScriptDir\gen_conv_model.py" `
  --two-layer `
  --output "$BuildDir\conv_model.onnx"

# Generate Conv+Gemm model
Write-Host ""
Write-Host "Generating Conv+Gemm model..."
python "$ScriptDir\gen_conv_gemm_model.py" `
  --output "$BuildDir\conv_gemm_model.onnx"

Write-Host ""
Write-Host "Models generated successfully in: $BuildDir"
Write-Host "  - conv_model.onnx (two-layer Conv)"
Write-Host "  - conv_gemm_model.onnx (Conv + Gemm)"
Write-Host ""
Write-Host "Run integration test with:"
Write-Host "  ctest --test-dir $BuildDir -R OrtIntegration --verbose"
