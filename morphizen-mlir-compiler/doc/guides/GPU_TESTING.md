<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GPU Testing Guide

Guide for testing ONNX HIP/DNN Execution Provider with AMD GPUs and TheRock ROCm SDK.

## Prerequisites

### Hardware
- AMD GPU with ROCm support (e.g., Radeon RX 6000/7000 series, Instinct MI series)

### Software
1. **ROCm Drivers**: Installed and functional
   ```bash
   # Verify ROCm installation
   rocminfo
   rocm-smi
   ```

2. **TheRock SDK**: Installed at `C:\Develop\m\dist\therock` (or custom location)

3. **Python Dependencies** (for comparison tests):
   ```bash
   pip install onnxruntime numpy
   ```

## Build Configuration

The project must be built with **real runtime** (not mock):

```bash
export THEROCK_DIST=C:/Develop/m/dist/therock
export HIP_PLATFORM=amd
LOCAL_DIR=$(cd ../../local && pwd)

cmake -S . -B ../../build/onnx-hipdnn-ep \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DONNX_MLIR_BUILD_TESTS=OFF \
  -DBUILD_MLIR_BACKEND=ON \
  -DBUILD_MLIR_HIP_OPT_TOOL=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh

cmake --build ../../build/onnx-hipdnn-ep --config Debug --parallel
```

## Testing Workflow

### Step 1: Compile Model to DLL

```bash
# Set environment
export THEROCK_DIST=C:/Develop/m/dist/therock

# Compile demo model
../../build/onnx-hipdnn-ep/Debug/bin/mlir-hip-compiler.exe \
  mlir-compiler/tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --from-onnx-mlir \
  -o ../output/demo_gpu.dll \
  --mode dll
```

**Expected output**:
```
=== Compilation Successful ===
Output: ../output/demo_gpu.dll
```

**Verify DLL**:
```bash
# Check exports
dumpbin //EXPORTS ../output/demo_gpu.dll | grep inference

# Should show:
#   inference_init
#   inference_compute
#   inference_cleanup

# Check dependencies
dumpbin //DEPENDENTS ../output/demo_gpu.dll

# Should show:
#   amdhip64_7.dll
#   MIOpen.dll
#   libhipblaslt.dll
#   KERNEL32.dll
```

### Step 2: Run Basic GPU Test

#### Windows (using batch script):
```cmd
cd C:\Develop\m\Source\onnx-hipdnn-ep
test_gpu_execution.bat
```

#### Manual execution:
```bash
# Add TheRock DLLs to PATH
export PATH="C:/Develop/m/dist/therock/bin:$PATH"

# Run test
../../build/onnx-hipdnn-ep/Debug/bin/test-model-dll.exe \
  ../output/demo_gpu.dll \
  --verbose
```

**Expected output (success)**:
```
=== Model DLL Test ===
DLL: ../output/demo_gpu.dll

--- Loading DLL ---
✓ DLL loaded successfully

--- Resolving Exports ---
✓ inference_init found
✓ inference_compute found
✓ inference_cleanup found

--- Running Inference ---
✓ inference_init succeeded
✓ inference_compute succeeded (XXX ms)
✓ inference_cleanup succeeded

--- Results ---
Output 0: [1, 64, 112, 112] float32
  Min: X.XXX, Max: X.XXX, Mean: X.XXX
```

### Step 3: GPU vs CPU Comparison

```bash
# Run comparison test
python mlir-compiler/test/e2e/test_gpu_vs_cpu.py \
  models/demo_two_layer_conv.onnx \
  --dll ../output/demo_gpu.dll \
  --tolerance 1e-4
```

**Expected output**:
```
==========================================================
GPU vs CPU Comparison Test
==========================================================
ONNX Model: models/demo_two_layer_conv.onnx
GPU DLL: ../output/demo_gpu.dll
Tolerance: 0.0001

Test input shape: (1, 3, 224, 224)

--- Running on CPU (ONNX Runtime) ---
✓ CPU execution succeeded
  Outputs: 1
  Output 0: shape=(1, 64, 112, 112), dtype=float32

--- Running on GPU (Compiled DLL) ---
✓ GPU execution succeeded
  Outputs: 1
  Output 0: shape=(1, 64, 112, 112), dtype=float32

--- Comparing Outputs ---
✓ Output 0: PASS
    Shape: (1, 64, 112, 112)
    Max diff: 1.23e-05
    Mean diff: 3.45e-06

==========================================================
SUCCESS: GPU and CPU outputs match!
==========================================================
```

## Troubleshooting

### Error 126: Cannot load DLL

**Cause**: Missing dependent DLL

**Solution**:
1. Ensure TheRock bin directory is in PATH:
   ```cmd
   set PATH=C:\Develop\m\dist\therock\bin;%PATH%
   ```

2. Verify all dependencies exist:
   ```bash
   ls C:/Develop/m/dist/therock/bin/{amdhip64_7.dll,MIOpen.dll,libhipblaslt.dll}
   ```

3. Check transitive dependencies:
   ```bash
   dumpbin //DEPENDENTS C:/Develop/m/dist/therock/bin/MIOpen.dll
   ```

   Required DLLs:
   - `hiprtc0702.dll`
   - `amd_comgr0702.dll`
   - `rocblas.dll`
   - `MSVCP140.dll` (Visual C++ Runtime)
   - `VCRUNTIME140.dll`

### HIP Error: hipErrorNoDevice

**Cause**: No AMD GPU found or ROCm drivers not installed

**Solution**:
1. Verify GPU is detected:
   ```bash
   rocm-smi
   rocminfo
   ```

2. Check device enumeration:
   ```bash
   hipconfig --version
   ```

3. Reinstall ROCm drivers if needed

### MIOpen Error: Empty code object path

**Cause**: MIOpen Find API called without workspace

**Solution**: Fixed in mlir-compiler/lib/Runtime/hipdnn_ep_runtime_miopen.cpp (lines 166-178). Allocates 10MB workspace before calling miopenFindConvolutionForwardAlgorithm.

### CRITICAL: TheRock SDK Windows DLL Context Bug

**Symptom**:
- Access violation / segmentation fault during `hipblasLtCreate()` initialization
- Error message: `gcnArchName is EMPTY (should be 'gfx1100' for W7900)`

**Cause**: PAL backend ISA query fails in DLL (LoadLibrary) context on Windows. The gcnArchName field returns empty, causing rocBLAS to crash with NULL pointer dereference.

**Workaround** - Use standalone .exe instead of DLL execution:

```bash
# ✅ THIS WORKS - Standalone executable
export PATH="C:/Develop/m/dist/therock/bin:$PATH"
../../build/onnx-hipdnn-ep/Debug/bin/test_miopen_then_hipblaslt.exe

# ❌ THIS CRASHES - DLL execution
../../build/onnx-hipdnn-ep/Debug/bin/test-model-dll.exe demo_two_layer.dll
```

**Note**: Setting `HSA_OVERRIDE_GFX_VERSION=11.0.0` does NOT fix DLL context issue.

**Recommended Solutions**:
1. **Short-term**: Use standalone .exe execution model (embed compiled model in executable)
2. **Medium-term**: Report to ROCm/TheRock maintainers at https://github.com/ROCm/TheRock/issues
3. **Long-term**: Debug PAL backend in ROCclr (`clr/hipamd/src/hip_device.cpp`)

### rocBLAS Error: Could not initialize Tensile host - No devices found

**Cause**: HIP runtime cannot enumerate AMD GPU compute devices (different from gcnArchName bug above)

**Possible Causes**:
1. TheRock SDK version mismatch with installed ROCm drivers
2. GPU in graphics mode only (not compute mode)
3. Missing HIP environment variables

**Investigation Steps**:
1. Check ROCm driver version matches TheRock SDK (7.2.x):
   ```bash
   C:/Develop/m/dist/therock/bin/hipconfig.exe --version
   rocm-smi --showdriverversion
   ```

2. Set HIP device visibility:
   ```cmd
   set HIP_VISIBLE_DEVICES=0
   set GPU_DEVICE_ORDINAL=0
   ```

3. Enable HIP debug logging:
   ```cmd
   set AMD_LOG_LEVEL=4
   ```

### Numerical Differences Between GPU and CPU

**Acceptable**:
- Relative difference < 1e-4 (due to floating-point precision)
- Different operation order (GPU parallelism)

**Investigate if**:
- Difference > 1e-3
- NaN or Inf values in output
- Completely different results

**Debug steps**:
1. Reduce tolerance to isolate precision issues
2. Compare intermediate layer outputs
3. Check if input data is being transferred correctly
4. Verify constant weights match

## Performance Benchmarking

```bash
# Run multiple iterations
../../build/onnx-hipdnn-ep/Debug/bin/test-model-dll.exe \
  ../output/demo_gpu.dll \
  --iterations 100 \
  --verbose
```

**Expected metrics**:
- First run: XXX ms (includes GPU initialization)
- Subsequent runs: XX ms (warm cache)

## Build Artifacts Verification

After successful build, verify these artifacts exist:

```
../../build/onnx-hipdnn-ep/
├── mlir-compiler/lib/Runtime/runtime.bc         # Real runtime (40KB, not mock)
├── Debug/bin/mlir-hip-compiler.exe              # Compiler tool (262MB)
├── Debug/bin/test-model-dll.exe                 # Test harness (2.5MB)
└── (generated)
    ../output/demo_gpu.dll                       # Compiled model (1.1MB)
```

**Verify runtime.bc is real (not mock)**:
```bash
# Should show 36+ references to HIP/MIOpen functions
C:/Develop/m/local/bin/llvm-dis.exe \
  ../../build/onnx-hipdnn-ep/mlir-compiler/lib/Runtime/runtime.bc -o - | \
  grep -c "hipMalloc\|miopenConvolution\|hipblasLt"
```

## CI/CD Integration

For automated testing without GPU:

1. **Build Verification** (no GPU needed):
   ```bash
   # Verify compilation succeeds
   cmake --build ../../build/onnx-hipdnn-ep --target mlir-hip-compiler

   # Verify DLL generation succeeds
   mlir-hip-compiler test.mlir -o test.dll

   # Verify DLL exports
   dumpbin //EXPORTS test.dll | grep inference
   ```

2. **GPU Tests** (requires GPU):
   - Run on dedicated GPU CI runner
   - Tag tests with `[gpu]` label
   - Allow longer timeout for first-run compilation

## Known Limitations

1. **No GPU Available**: Tests will fail if no AMD GPU or ROCm drivers installed
2. **TheRock SDK Required**: Real runtime needs TheRock ROCm libraries
3. **Windows Only**: Current DLL linking setup is Windows-specific (LLD-LINK)
4. **Float32 Only**: Current runtime only supports float32 data type
5. **Static Input Shapes**: Compiled models have fixed input shapes
6. **DLL Context Bug**: TheRock SDK on Windows has PAL backend ISA query failure in DLL context - use standalone .exe execution

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [MIOpen API Reference](https://rocm.docs.amd.com/projects/MIOpen/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- Project Architecture: `doc/ARCHITECTURE.md`
- Runtime Design: `doc/RUNTIME-ARCHITECTURE.md`
