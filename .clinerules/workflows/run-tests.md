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

## Step 3: Optional - Enable Debug Output

For detailed debug information:

```powershell
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
[ RUN      ] OrtIntegrationTest.LoadMorphiZenProvider
[       OK ] OrtIntegrationTest.LoadMorphiZenProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithMorphiZenProvider
[       OK ] OrtIntegrationTest.CreateSessionWithMorphiZenProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[       OK ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[----------] 3 tests from OrtIntegrationTest
[  PASSED  ] 3 tests.
```

## Test Cases

### 1. LoadMorphiZenProvider
Verifies MorphiZen EP can be loaded and registered successfully.

### 2. CreateSessionWithMorphiZenProvider
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

# Run tests
..\build\onnx-hipdnn-ep\bin\Release\ort_integration_test.exe
```

## Additional Resources

For comprehensive testing documentation, see [doc/TESTING.md](../doc/TESTING.md).
