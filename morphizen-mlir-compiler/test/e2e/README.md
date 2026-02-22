<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# E2E Tests - Quick Guide

## Prerequisites

1. ONNX Runtime and GTest installed in `../../local/`
2. Build morphizen-mlir-compiler with `BUILD_MOCK_RUNTIME=ON`
3. Generate test models (see below)

## Generate Test Models

```bash
# From morphizen-mlir-compiler/test/e2e/
python gen_conv_model.py --two-layer --output /c/Develop/m/build/onnx-hipdnn-ep-mlir-integration-1/Debug/bin/conv_model.onnx
python gen_conv_gemm_model.py --output /c/Develop/m/build/onnx-hipdnn-ep-mlir-integration-1/Debug/bin/conv_gemm_model.onnx
```

## Run ORT Integration Test

```bash
cd /c/Develop/m/build/onnx-hipdnn-ep-mlir-integration-1/Debug/bin

# Basic test
MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1 ./ort_integration_test.exe

# With verbose logging
ORT_LOG_LEVEL=info \
DEBUG_LOG_LEVEL=info \
XLNX_ONNX_EP_VERBOSE=2 \
DEBUG_MORPHIZEN_PASS=1 \
MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1 \
./ort_integration_test.exe

# Run specific test
MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1 \
./ort_integration_test.exe --gtest_filter=OrtIntegrationTest.CreateSessionWithMorphiZenProvider
```

## Environment Variables

- `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1` - **Required**: Enable CPU device for MorphiZen EP
- `ORT_LOG_LEVEL=info` - ONNX Runtime logging (session creation, graph transforms)
- `DEBUG_LOG_LEVEL=info` - General debug logging
- `XLNX_ONNX_EP_VERBOSE=2` - Vitis AI EP verbose logging (cache, versions, statistics)
- `DEBUG_MORPHIZEN_PASS=1` - Enable morphizen pass debug logging
- `MORPHIZEN_DEBUG_MLIR_BACKEND=3` - MLIR backend compilation verbose logging

## Expected Output

```
[  PASSED  ] LoadMorphiZenProvider - EP registered
[  PASSED  ] CreateSessionWithMorphiZenProvider - Session created with Conv model
[  PASSED  ] CreateSessionWithConvGemmModel - Session created with Conv+Gemm model
```

## Test Models

- **conv_model.onnx**: Two-layer Conv [1,3,224,224] → [1,64,112,112]
- **conv_gemm_model.onnx**: Conv+Gemm [1,3,8,8] → [1,32]

## Logs and Artifacts

- **Cache dir**: `C:\temp\chunywan\vaip\.cache\<cache-key>/`
- **Init graph**: `./vaip_init/init-graph-for-requ-dq.onnx`
- **Log dir**: Set by `XLNX_ONNX_EP_VERBOSE` (in-mem mode: no persistent logs)

## Troubleshooting

**Test skips with "V2 API not implemented"**:
- Set `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1`

**Missing DLLs**:
The required DLLs should be automatically copied to the bin directory during build:
- `onnxruntime.dll` - From ../../local/bin (ONNX Runtime)
- `onnxruntime_providers_shared.dll` - From ../../local/bin (ONNX Runtime shared providers)
- `onnxruntime_morphizen_ep.dll` - **Built from 3rd-party/morphizen** (do NOT use the one from ../../local/bin, it's outdated!)

**CRITICAL - Wrong morphizen EP DLL**:
If you see old morphizen version in logs (e.g., version from Feb 17), ensure you're using the newly built DLL:
```bash
# The bin directory should contain the newly built morphizen EP DLL (from current build)
# Do NOT manually copy onnxruntime_morphizen_ep.dll from ../../local/bin (it's outdated)
ls -lh onnxruntime_morphizen_ep.dll  # Should show recent build timestamp
```

**API version mismatch**:
- The ONNX Runtime DLL (onnxruntime.dll) should be copied from ../../local/bin automatically during build
- If you see API version errors, verify the DLL in bin directory matches the one in ../../local/bin

See `README_E2E_TESTING.md` for detailed documentation.
