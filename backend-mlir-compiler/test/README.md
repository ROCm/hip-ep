# MLIR Backend E2E Tests

## Overview
End-to-end tests for MLIR backend integration. Works with both MOCK runtime (compile-time default) and REAL runtime (compile-time option with `BUILD_MOCK_RUNTIME=OFF`).

## Prerequisites
- ONNX Runtime installed in `../../local/`
- GTest available
- MorphiZen MLIR compiler built with `BUILD_MOCK_RUNTIME=ON` (default)

## Test Model
The test uses `models/two_layer_conv.onnx` (committed via Git LFS):
- Input: `[1, 3, 224, 224]` (batch=1, RGB, 224x224 image)
- Pipeline: Conv1 → ReLU → Conv2 → ReLU
- Output: `[1, 64, 112, 112]`

### Regenerating the Model (Optional)
If you need to regenerate the test model:
```bash
python gen_two_layer_conv_model.py --output models/two_layer_conv.onnx
git add models/two_layer_conv.onnx  # Git LFS will handle it automatically
```

## Build
```bash
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S ../.. -B ../../build/onnx-hipdnn-ep-mlir-integration \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  --fresh

cmake --build ../../build/onnx-hipdnn-ep-mlir-integration --config Debug --parallel
```

## Running the Test

### With MOCK Runtime (default, no GPU required)
```bash
cd ../../build/onnx-hipdnn-ep-mlir-integration/Debug/bin

# Basic run
./mlir_e2e_test.exe

# With verbose logging
ORT_LOG_LEVEL=info \
DEBUG_MORPHIZEN_PASS=1 \
MORPHIZEN_DEBUG_MLIR_BACKEND=3 \
./mlir_e2e_test.exe

# Using ctest
cd ../../build/onnx-hipdnn-ep-mlir-integration
ctest -R MlirE2ETest --verbose
```

### With REAL Runtime (requires ROCm GPU)
```bash
# Rebuild with REAL runtime
cmake -S . -B ../../build/onnx-hipdnn-ep-mlir-integration \
  -DBUILD_MOCK_RUNTIME=OFF \
  ... (other options)

cmake --build ../../build/onnx-hipdnn-ep-mlir-integration --config Debug

# Run test
cd ../../build/onnx-hipdnn-ep-mlir-integration/Debug/bin
./mlir_e2e_test.exe
```

## Expected Output

### MOCK Runtime
```
[==========] Running 1 test from 1 test suite.
[----------] 1 test from MlirE2ETest
[ RUN      ] MlirE2ETest.TwoLayerConvSession
[SetUp] MorphiZen EP registered successfully
[Test] Creating session with MorphiZen EP (MLIR backend)...
[MOCK] hipGetDeviceCount
[MOCK] hipStreamCreate() -> <address>
[MOCK] wrap_miopenConvolutionForward(...)
[Test] Session created successfully with MorphiZen EP!
[  PASSED  ] MlirE2ETest.TwoLayerConvSession
[==========] 1 test from 1 test suite ran.
[  PASSED  ] 1 test.
```

### REAL Runtime (TODO)
```
[Test] Session created successfully with MorphiZen EP!
[GPU] Convolution executed on device
[  PASSED  ] MlirE2ETest.TwoLayerConvSession
```

## Environment Variables (All Optional)
- `ORT_LOG_LEVEL=info` - Enable ORT session creation logging
- `DEBUG_MORPHIZEN_PASS=1` - Enable morphizen pass debug logging
- `MORPHIZEN_DEBUG_MLIR_BACKEND=3` - MLIR backend compilation verbose logging

**Note**: CPU device support is enabled by default in MorphiZen EP (`MORPHIZEN_EP_ENABLE_CPU_DEVICE=1`), so no special environment variables are required to run the test.

## TODO - Future Enhancements
- Add actual inference testing (forward pass with input data, output validation)
- Test with REAL runtime on GPU hardware
- Add output validation (compare MOCK zeros vs REAL computed results)
- Add performance benchmarking for REAL runtime

## File Structure
```
backend-mlir-compiler/test/
├── CMakeLists.txt               # Build configuration
├── README.md                    # This file
├── gen_two_layer_conv_model.py  # Model generation script
├── models/                      # Test models (Git LFS)
│   └── two_layer_conv.onnx      # Two-layer convolution model
└── test_e2e_mlir.cpp            # E2E test implementation
```
