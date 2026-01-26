<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.
-->
# Run Tests

Run the ORT integration tests for the onnx-hipdnn-ep project.

## Prerequisites

1. **Build Complete**: Ensure the project is built successfully (see `build-and-fix-build-errors.md`)
2. **Test Models Generated**: Test ONNX models must be generated and copied to the build output directory

## Step 1: Generate Test Models

From the `onnx-hipdnn-ep` directory:

```powershell
cd test
python gen_conv_model.py
python gen_conv_gemm_model.py
cd ..
```

This generates:
- `test/conv_model.onnx` - Simple Conv operation model
- `test/conv_gemm_model.onnx` - Conv + Flatten + Gemm pipeline model

## Step 2: Copy Models to Build Output

```powershell
Copy-Item test\*.onnx ..\build\onnx-hipdnn-ep\bin\Release\
```

## Step 3: Set Required Environment Variables

**CRITICAL:** Set the environment variable to enable CPU device testing:

```powershell
$env:MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE = "1"
```

**Why this is required:** The VitisAI EP normally requires NPU hardware. This variable enables CPU device mode for testing without NPU.

### Optional: Enable Debug Output

For detailed debug information:

```powershell
$env:MORPHIZEN_DEBUG_MLIR = "2"
$env:MORPHIZEN_DEBUG_MLIR_GRAPH = "2"
$env:GLOG_logtostderr = "1"
$env:GLOG_minloglevel = "0"
```

## Step 4: Run Tests

```powershell
..\build\onnx-hipdnn-ep\bin\Release\ort_integration_test.exe
```

## Expected Output

Successful test run should show:

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from OrtIntegrationTest
[ RUN      ] OrtIntegrationTest.LoadVitisAIProvider
[       OK ] OrtIntegrationTest.LoadVitisAIProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithVitisAIProvider
[       OK ] OrtIntegrationTest.CreateSessionWithVitisAIProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[       OK ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[----------] 3 tests from OrtIntegrationTest
[  PASSED  ] 3 tests.
```

## Test Cases

### 1. LoadVitisAIProvider
Verifies VitisAI EP can be loaded and registered successfully.

### 2. CreateSessionWithVitisAIProvider
Tests session creation with Conv model (`conv_model.onnx`):
- Input: [1, 3, 8, 8]
- Output: [1, 16, 8, 8]

### 3. CreateSessionWithConvGemmModel
Tests session creation with Conv+Gemm model (`conv_gemm_model.onnx`):
- Input: [1, 3, 8, 8]
- Output: [1, 32]

## Troubleshooting

### Test Models Not Found
**Error:** "Model file not found"
**Fix:** Run Step 1 and Step 2 to generate and copy models

### Tests Skipped
**Error:** "VitisAI EP V2 device API not yet implemented"
**Fix:** Set `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1` environment variable (Step 3)

### DLL Not Found
**Error:** "Cannot find onnxruntime.dll"
**Fix:** Add DLL paths to PATH:
```powershell
$env:PATH = "..\local\bin;..\build\onnx-hipdnn-ep\bin\Release;$env:PATH"
```

### Test Executable Not Found
**Error:** Test executable doesn't exist
**Fix:** Build the project first (see `build-and-fix-build-errors.md`)

## Quick All-in-One Command

From `onnx-hipdnn-ep` directory:

```powershell
# Generate and copy models
cd test; python gen_conv_model.py; python gen_conv_gemm_model.py; cd ..
Copy-Item test\*.onnx ..\build\onnx-hipdnn-ep\bin\Release\

# Set environment variable
$env:MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE = "1"

# Run tests
..\build\onnx-hipdnn-ep\bin\Release\ort_integration_test.exe
```

## Additional Resources

For comprehensive testing documentation, see [doc/TESTING.md](../doc/TESTING.md).
