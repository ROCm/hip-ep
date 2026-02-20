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
cmake -S ../.. -B ../../build/$(basename $PWD) \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  --fresh

cmake --build ../../build/$(basename $PWD) --config Debug --parallel
```

## Running the Test

### With MOCK Runtime (default, no GPU required)
```bash
cd ../../build/$(basename $PWD)/Debug/bin

# Basic run
./mlir_e2e_test.exe

# With verbose logging
ORT_LOG_LEVEL=info \
DEBUG_MORPHIZEN_PASS=1 \
MORPHIZEN_DEBUG_MLIR_BACKEND=3 \
./mlir_e2e_test.exe

# Using ctest
cd ../../build/$(basename $PWD)
ctest -R MlirE2ETest --verbose
```

### With REAL Runtime (requires ROCm GPU)
```bash
# Rebuild with REAL runtime
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S ../.. -B ../../build/$(basename $PWD) \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh

cmake --build ../../build/$(basename $PWD) --config Debug --parallel

# Run test
cd ../../build/$(basename $PWD)/Debug/bin
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

## Environment Variables

The test automatically sets these environment variables for verbose logging:
- `XLNX_ONNX_EP_VERBOSE=2` - Enable ONNX EP verbose logging
- `DEBUG_LOG_LEVEL=info` - Enable debug logging
- `MORPHIZEN_DEBUG_PLUGIN=1` - Enable plugin loading debug logging

Additional optional environment variables:
- `ORT_LOG_LEVEL=info` - Enable ORT session creation logging (configurable)
- `DEBUG_MORPHIZEN_PASS=1` - Enable morphizen pass debug logging
- `MORPHIZEN_DEBUG_MLIR_BACKEND=3` - MLIR backend compilation verbose logging
- `MORPHIZEN_DEBUG_MLIR_BACKEND=2` - Dump MLIR bytecode to `<dump_dir>/mlir_bytecode_dump.mlir`

**Note**: CPU device support is enabled by default in MorphiZen EP (`MORPHIZEN_EP_ENABLE_CPU_DEVICE=1`), so no special environment variables are required to run the test.

**Troubleshooting**: See the "Troubleshooting MLIR Compilation Errors" section for using `hip-opt --all-passes` and MLIR debugging flags.

## Troubleshooting MLIR Compilation Errors

If the test shows MLIR compilation failures, use this workflow to debug:

### 1. Enable Bytecode Dumping

Set `MORPHIZEN_DEBUG_MLIR_BACKEND=2` to dump the MLIR bytecode to file:

```bash
# Navigate to build output directory
cd ../../build/$(basename $PWD)/Debug/bin

MORPHIZEN_DEBUG_MLIR_BACKEND=2 \
XLNX_ONNX_EP_VERBOSE=2 \
DEBUG_LOG_LEVEL=info \
MORPHIZEN_DEBUG_PLUGIN=1 \
./mlir_e2e_test.exe
```

This creates `mlir_bytecode_dump.mlir` in the dump directory. Check the test output for the dump directory path:

```
[XLNX_ONNX_EP_VERBOSE] EXEC VERSION: dump_dir: "<dump_directory_path>"
```

The bytecode file will be at: `<dump_directory>/mlir_bytecode_dump.mlir`

On Windows, the default dump directory is typically `C:\temp\morphizen_dumps\<cache_key>\`.

### 2. Locate the Bytecode Dump

Find the dump directory from the test output and navigate to it:

```bash
# Extract dump directory from test output (look for "dump_dir:" line)
# Windows Git Bash: DUMP_DIR="/c/temp/morphizen_dumps/<cache_key>"
# Linux: DUMP_DIR="/tmp/morphizen_dumps/<cache_key>"

# Navigate to dump directory (replace <cache_key> with actual value from test output)
cd "$DUMP_DIR"
ls -lh mlir_bytecode_dump.mlir
```

### 3. Convert Bytecode to Text Format

Use `mlir-opt` to convert the binary bytecode to human-readable MLIR text:

```bash
# Store the dump directory path
DUMP_DIR="<path_from_test_output>"

# mlir-opt is in ../../local/bin/ relative to project root
# Assuming you're in the project root directory:
LOCAL_DIR=$(cd ../../local && pwd)

"$LOCAL_DIR/bin/mlir-opt" --allow-unregistered-dialect \
  "$DUMP_DIR/mlir_bytecode_dump.mlir" \
  -o two_layer_conv.mlir
```

### 4. Save Test File for Reference

Copy the converted MLIR to the morphizen-mlir-compiler test directory:

```bash
# From project root directory
mkdir -p 3rd-party/morphizen/morphizen-mlir-compiler/test/e2e

# Copy the converted MLIR
cp two_layer_conv.mlir \
  3rd-party/morphizen/morphizen-mlir-compiler/test/e2e/two_layers_conv_real_constant.mlir
```

### 5. Test Complete Pipeline with hip-opt

Use `hip-opt` with `--all-passes` to test the full ONNX→HIP→LLVM→Interface pipeline:

```bash
# From project root directory
BUILD_DIR=$(cd ../../build/$(basename $PWD) && pwd)

"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  two_layer_conv.mlir \
  -o two_layer_conv_compiled.mlir
```

The `--all-passes` flag runs the complete compilation pipeline (equivalent to the passes in `CompilerPipeline::runMLIRPasses()`):
1. `--convert-onnx-to-hip` - Convert ONNX operations to HIP dialect
2. `--func.func='buffer-loop-hoisting,ownership-based-buffer-deallocation,optimize-allocation-liveness'` - Buffer management passes (nested under func.func)
3. `--canonicalize` - Canonicalization pass
4. `--hip-memory-pooling` - Memory pooling optimization
5. `--convert-hip-to-llvm` - Convert HIP dialect to LLVM dialect
6. `--hip-generate-interface` - Generate C interface wrapper functions

If successful, this validates that all MLIR transformations work correctly.

### 6. Debug Individual Passes with MLIR Built-in Flags

MLIR's pass manager has built-in debugging flags to dump IR before/after each pass. Use these to identify which pass is failing:

```bash
# Dump IR before ALL passes
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-before-all \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 | tee debug_all_before.log

# Dump IR after ALL passes
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 | tee debug_all_after.log

# Dump IR before a SPECIFIC pass (e.g., convert-onnx-to-hip)
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-before=convert-onnx-to-hip \
  --mlir-print-ir-module-scope \
  two_layer_conv.mlir 2>&1 | tee debug_before_onnx_to_hip.log

# Dump IR after a SPECIFIC pass (e.g., convert-onnx-to-hip)
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-after=convert-onnx-to-hip \
  --mlir-print-ir-module-scope \
  two_layer_conv.mlir 2>&1 | tee debug_after_onnx_to_hip.log
```

**Available MLIR Debug Flags**:
- `--mlir-print-ir-before=<pass-name>` - Print IR before a specific pass
- `--mlir-print-ir-after=<pass-name>` - Print IR after a specific pass
- `--mlir-print-ir-before-all` - Print IR before all passes
- `--mlir-print-ir-after-all` - Print IR after all passes
- `--mlir-print-ir-module-scope` - Always print at module scope (recommended for readability)
- `--mlir-disable-threading` - Disable multi-threading for deterministic output (recommended for debugging)

**Pass names** (use with `--mlir-print-ir-before=<name>` or `--mlir-print-ir-after=<name>`):
- `convert-onnx-to-hip` - ONNX to HIP conversion
- `buffer-loop-hoisting` - Buffer loop hoisting (nested pass)
- `ownership-based-buffer-deallocation` - Buffer deallocation (nested pass)
- `optimize-allocation-liveness` - Allocation liveness optimization (nested pass)
- `canonicalize` - Canonicalization
- `hip-memory-pooling` - Memory pooling
- `convert-hip-to-llvm` - HIP to LLVM conversion
- `hip-generate-interface` - Interface generation

### 7. Isolate the Problematic Pass (Two Methods)

Once you identify the failing pass from the logs, use one of these methods to isolate it:

#### Method 1: Use --mlir-print-ir-tree-dir (Recommended)

MLIR can automatically dump IR to files before/after each pass:

```bash
# Create output directory for IR dumps
mkdir -p ir_dumps

# Run with automatic IR file dumping
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-tree-dir=ir_dumps \
  --mlir-print-ir-before=convert-hip-to-llvm \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 | tee debug.log

# IR files are created in ir_dumps/ with timestamps
# Example: ir_dumps/pipeline_convert-hip-to-llvm_before_0.mlir
ls -lt ir_dumps/
```

**Advantages:**
- Automatic file creation with meaningful names
- No manual extraction needed
- Timestamps for tracking multiple runs

#### Method 2: Manual Extraction from Logs

If `--mlir-print-ir-tree-dir` is not available or you prefer manual control:

```bash
# 1. Dump IR before failing pass to stdout
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-before=convert-hip-to-llvm \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 > debug_before.log

# 2. Extract the IR from the log (between IR dump markers)
# Look for "// -----// IR Dump Before ConvertHipToLLVM //" in the log
# Copy the MLIR code block to a new file
cat debug_before.log | sed -n '/IR Dump Before/,/^$/p' > input_to_failing_pass.mlir

# Or manually extract using a text editor
```

#### Method 3: Run Passes Incrementally

Run passes one-by-one up to the failing pass:

```bash
# Run only passes BEFORE the failing pass
# Example: If convert-hip-to-llvm fails, run up to hip-memory-pooling

"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-onnx-to-hip \
  --func.func='buffer-loop-hoisting,ownership-based-buffer-deallocation,optimize-allocation-liveness' \
  --canonicalize \
  --hip-memory-pooling \
  -o input_before_failing_pass.mlir \
  two_layer_conv.mlir

# Now test the failing pass in isolation
"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-hip-to-llvm \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  input_before_failing_pass.mlir 2>&1 | tee isolated_pass_debug.log
```

**When to use each method:**
- **Method 1**: Best for automatic troubleshooting, clean workflow
- **Method 2**: When you need full control over extracted IR
- **Method 3**: When you want to test pass combinations or dependencies

### 8. Run Isolated Pass for Root Cause Analysis

Once you have the input IR file (from any method above):

```bash
# Run the specific failing pass with verbose output
"$BUILD_DIR/bin/Debug/hip-opt" \
  --<failing-pass-name> \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  input_to_failing_pass.mlir 2>&1 | tee isolated_pass_debug.log

# Example for convert-hip-to-llvm:
"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-hip-to-llvm \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  input_before_convert_hip_to_llvm.mlir 2>&1 | tee hip_to_llvm_isolated.log
```

This narrows down the root cause to a specific transformation with minimal noise.

### 9. Complete Example: Debugging a Failing Pass

Here's a complete reproducible workflow from start to finish:

```bash
# Setup (from project root)
BUILD_DIR=$(cd ../../build/$(basename $PWD) && pwd)
cd backend-mlir-compiler/test

# Step 1: Run test to get bytecode dump
MORPHIZEN_DEBUG_MLIR_BACKEND=2 \
  "$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"

# Step 2: Find dump directory from test output
# Look for: dump_dir: "C:\\temp\\morphizen_dumps\\<cache_key>"
DUMP_DIR="/c/temp/morphizen_dumps/<cache_key>"

# Step 3: Convert bytecode to text
LOCAL_DIR=$(cd ../../local && pwd)
"$LOCAL_DIR/bin/mlir-opt" --allow-unregistered-dialect \
  "$DUMP_DIR/mlir_bytecode_dump.mlir" \
  -o two_layer_conv.mlir

# Step 4: Test complete pipeline to identify failing pass
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-after-all \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 | tee full_pipeline_debug.log

# Step 5: If a pass fails (e.g., convert-hip-to-llvm), dump IR before it
mkdir -p ir_dumps
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-tree-dir=ir_dumps \
  --mlir-print-ir-before=convert-hip-to-llvm \
  --mlir-disable-threading \
  two_layer_conv.mlir 2>&1 | tee debug_with_dumps.log

# Step 6: Find the dumped IR file
# Files are named like: pipeline_convert-hip-to-llvm_before_0.mlir
IR_FILE=$(ls -t ir_dumps/pipeline_*before*.mlir | head -1)
echo "Input IR: $IR_FILE"

# Step 7: Run failing pass in isolation
"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-hip-to-llvm \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  "$IR_FILE" 2>&1 | tee isolated_pass_debug.log

# Step 8: Analyze the isolated output for root cause
# Look for pattern matching failures, type mismatches, etc.
grep -E "(error|failed|mismatch)" isolated_pass_debug.log
```

### 10. Common Issues

**Malformed null-terminated string error**:
- This occurred when `from_onnx_mlir = false` was hardcoded, causing all passes to be skipped
- Fixed by removing the `from_onnx_mlir` flag and always running the full pipeline

**Unknown dialect errors**:
- Ensure all required MLIR dialects are registered (ONNX, HIP, LLVM)
- Check that onnx-mlir is built correctly

**Pass failures**:
- Use `--mlir-print-ir-before=<pass-name>` and `--mlir-print-ir-after=<pass-name>` to isolate which pass is failing
- Extract the input IR and run the specific pass in isolation
- Check the MLIR IR at each stage

**Pattern matching failures in ONNX→HIP conversion**:
- Review the ONNX operations in the input IR
- Check if all required ONNX ops have HIP lowering patterns registered
- Use `--mlir-print-ir-after=convert-onnx-to-hip` to verify the conversion succeeded

## TODO - Future Enhancements
- Add actual inference testing (forward pass with input data, output validation)
- Test with REAL runtime on GPU hardware
- Add output validation (compare MOCK zeros vs REAL computed results)
- Add performance benchmarking for REAL runtime
- Fix MLIR bytecode parsing issue ("malformed null-terminated string")

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
