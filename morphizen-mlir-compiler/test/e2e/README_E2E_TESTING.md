<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# E2E ONNX Integration Testing

This document describes how to generate ONNX test models and run E2E integration tests for the MorphiZen MLIR backend.

## Quick Start

### Generate Test Models

```bash
# Windows (PowerShell)
cd mlir-compiler/test/e2e
powershell -File generate_test_models.ps1

# Linux/Mac (Bash)
cd mlir-compiler/test/e2e
./generate_test_models.sh
```

This generates two ONNX models in the build directory:
- `conv_model.onnx`: Two-layer Conv model (similar to demo_two_layer_conv.mlir)
- `conv_gemm_model.onnx`: Conv + Flatten + Gemm model

### Run Integration Test

```bash
# Build the test first
cmake --build ../../build/onnx-hipdnn-ep-mlir-integration-1 --target ort_integration_test --config Debug

# Run the test
cd ../../build/onnx-hipdnn-ep-mlir-integration-1
./Debug/bin/ort_integration_test.exe
```

## Model Details

### Two-Layer Conv Model (`conv_model.onnx`)

Generated with `--two-layer` flag, this model matches the structure of `demo_two_layer_conv.mlir`:

**Pipeline:**
```
Input [1, 3, 224, 224]
  -> Conv1 (3->64 channels, 3x3 kernel, stride=1, pad=1)
  -> Conv1 output: [1, 64, 224, 224]
  -> ReLU1
  -> Conv2 (64->64 channels, 3x3 kernel, stride=2, pad=1)
  -> Conv2 output: [1, 64, 112, 112]
  -> ReLU2
  -> Output [1, 64, 112, 112]
```

**Key Features:**
- First Conv: Same spatial dimensions (stride=1)
- Second Conv: Halves spatial dimensions (stride=2)
- ReLU activations after each Conv
- Matches demo MLIR structure for ONNX → MLIR → HIP pipeline testing

### Conv+Gemm Model (`conv_gemm_model.onnx`)

**Pipeline:**
```
Input [1, 3, 8, 8]
  -> Conv (3->16 channels, 3x3 kernel, stride=1, pad=1)
  -> Conv output: [1, 16, 8, 8]
  -> Flatten
  -> Flatten output: [1, 1024]
  -> Gemm (1024->32)
  -> Output [1, 32]
```

**Key Features:**
- Tests Conv + Gemm operation fusion
- Smaller input size (8x8) for faster testing
- Tests Flatten operation between Conv and Gemm

## Manual Model Generation

### Two-Layer Conv Model

```bash
python gen_conv_model.py \
  --two-layer \
  --batch 1 \
  --in-channels 3 \
  --height 224 \
  --width 224 \
  --output ../../build/onnx-hipdnn-ep-mlir-integration-1/conv_model.onnx
```

### Single-Layer Conv Model

```bash
python gen_conv_model.py \
  --batch 1 \
  --in-channels 3 \
  --height 8 \
  --width 8 \
  --out-channels 16 \
  --kernel 3 \
  --padding 1 \
  --stride 1 \
  --bias \
  --output conv_single.onnx
```

### Conv+Gemm Model

```bash
python gen_conv_gemm_model.py \
  --batch 1 \
  --in-channels 3 \
  --height 8 \
  --width 8 \
  --out-channels 16 \
  --kernel 3 \
  --padding 1 \
  --stride 1 \
  --gemm-n 32 \
  --output ../../build/onnx-hipdnn-ep-mlir-integration-1/conv_gemm_model.onnx
```

## Verification

### Inspect ONNX Model Structure

```python
import onnx

model = onnx.load('path/to/model.onnx')

print('Graph:')
print(f'  Inputs: {[i.name for i in model.graph.input]}')
print(f'  Outputs: {[o.name for o in model.graph.output]}')

print('\nNodes:')
for node in model.graph.node:
    print(f'  {node.op_type} ({node.name}): {list(node.input)} -> {list(node.output)}')
```

## Integration Test Details

The ORT integration test (`test_ort_integration.cpp`) verifies:

1. **LoadMorphiZenProvider**: MorphiZen EP loads successfully
2. **CreateSessionWithMorphiZenProvider**: Session created with Conv model
3. **CreateSessionWithConvGemmModel**: Session created with Conv+Gemm model

**Test Behavior:**
- Tests skip gracefully if MorphiZen EP or models are unavailable
- Tests verify MLIR passes execute during session creation
- Tests check for compilation errors in MLIR → HIP pipeline

## Known Issues

### API Version Mismatch

The test may show an API version warning:
```
The requested API version [24] is not available, only API versions [1, 17] are supported
```

This indicates the test was built with a newer ONNX Runtime API but is running against an older version. The test infrastructure and model generation still work correctly.

### MorphiZen EP Not Available

If the MorphiZen EP DLL is not found, tests will skip gracefully with message:
```
MorphiZen EP not available: <error message>
```

## File Locations

- **Model generators**: `mlir-compiler/test/e2e/gen_conv_model.py`, `gen_conv_gemm_model.py`
- **Helper scripts**: `mlir-compiler/test/e2e/generate_test_models.{sh,ps1}`
- **Integration test**: `mlir-compiler/test/e2e/test_ort_integration.cpp`
- **Generated models**: `../../build/onnx-hipdnn-ep-mlir-integration-1/*.onnx` (not committed)
- **Test executable**: `../../build/onnx-hipdnn-ep-mlir-integration-1/Debug/bin/ort_integration_test.exe`

## Notes

- Generated ONNX models are **not committed** to the repository (in `.gitignore`)
- Models must be regenerated after cleaning the build directory
- The two-layer Conv model is designed to match `demo_two_layer_conv.mlir` for testing the ONNX → MLIR → HIP pipeline
- Model generation is separate from the build process (manual step)
