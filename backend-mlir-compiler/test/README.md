# MLIR Backend E2E Tests

## Overview
End-to-end tests for MLIR backend integration. Works with both MOCK runtime (compile-time default) and REAL runtime (compile-time option with `BUILD_MOCK_RUNTIME=OFF`).

## Directory Structure

This README assumes the following structure:
```
<workspace>/                       # Two levels up from project root
├── local/                         # Install prefix (LOCAL_DIR = $PROJECT_ROOT/../../local)
│   ├── bin/                       # MLIR tools (mlir-opt, hip-opt, etc.)
│   ├── lib/                       # Libraries
│   └── ...
└── onnx-hipdnn-ep/
    └── mlir-integration/          # Project root (PROJECT_ROOT)
        ├── backend-mlir-compiler/
        │   └── test/              # You are here
        │       ├── README.md      # This file
        │       ├── test_e2e_mlir.cpp
        │       ├── models/two_layer_conv.onnx
        │       └── ...
        └── build/                 # Build output (gitignored)
            └── test/              # Build directory (BUILD_DIR = $PROJECT_ROOT/build/test)
```

**Note**: The `local/` directory is at `$PROJECT_ROOT/../../local` (two levels up from project root). If your dependencies are installed elsewhere, set `CMAKE_PREFIX_PATH` to that location.

**CRITICAL - Debugging Workflow**:
- NEVER save temporary MLIR files in the project workspace
- Use temp directories for debugging:
  - Windows: `/c/temp/` (Git Bash) or `C:\temp\` (CMD)
  - Linux: `/tmp/`
- Once root cause is identified, create LIT unit tests in `3rd-party/morphizen/morphizen-mlir-compiler/test/lit/`
- See "Creating LIT Unit Tests from E2E Failures" section below

## Prerequisites
- ONNX Runtime (via CMAKE_PREFIX_PATH)
- LLVM/MLIR (via CMAKE_PREFIX_PATH or auto-fetched)
- GTest (auto-fetched if not found)
- Protobuf (auto-fetched if not found)

## Test Model
The test uses `models/two_layer_conv.onnx` (committed via Git LFS):
- Input: `[1, 3, 224, 224]` (batch=1, RGB, 224x224 image)
- Pipeline: Conv1 → ReLU → Conv2 → ReLU
- Output: `[1, 64, 112, 112]`

### Regenerating the Model (Optional)
If you need to regenerate the test model:
```bash
# From backend-mlir-compiler/test directory
python gen_two_layer_conv_model.py --output models/two_layer_conv.onnx
git add models/two_layer_conv.onnx  # Git LFS will handle it automatically
```

## Build

### Step 1: Set Up Directory Variables

**CRITICAL**: All commands in this guide run from the project root. Never use `cd` commands except the initial navigation to the project root.

```bash
# Navigate to project root ONCE
cd /c/Develop/m/onnx-hipdnn-ep/mlir-integration  # Adjust to your path

# Set up directory variables
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build/test"
LOCAL_DIR="$PROJECT_ROOT/../../local"  # Two levels up from project root
TEMP_DIR="/c/temp"  # Windows Git Bash
# TEMP_DIR="/tmp"   # Linux
```

### Step 2: Configure CMake
```bash
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  --fresh
```

### Step 3: Build
```bash
cmake --build "$BUILD_DIR" --config Debug --parallel
```

## Running the Test

### With MOCK Runtime (default, no GPU required)

```bash
# Basic run
"$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"

# With verbose logging
ORT_LOG_LEVEL=info \
DEBUG_MORPHIZEN_PASS=1 \
MORPHIZEN_DEBUG_MLIR_BACKEND=3 \
"$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"

# Using ctest
ctest --test-dir "$BUILD_DIR" -R MlirE2ETest --verbose
```

### With REAL Runtime (requires ROCm GPU)
```bash
# Rebuild with REAL runtime
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh

cmake --build "$BUILD_DIR" --config Debug --parallel

# Run test
"$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"
```

### Troubleshooting: Missing DLL Dependencies

If you encounter an error loading `morphizen-mlir-compiler.dll` due to missing dependencies:

```bash
# Workaround: Copy DLLs from local/bin to the binary directory
cp "$LOCAL_DIR/bin"/*.dll "$BUILD_DIR/Debug/bin/"
```

This copies all required DLLs from the install directory to the test binary directory.

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

When the E2E test shows MLIR compilation failures, follow this **evidence-based systematic workflow**. The key principle: **use debug flags to gather evidence, not speculation**.

### Prerequisites: Set Up Environment

Run ALL commands from the project root:

```bash
# Navigate to project root ONCE
cd /c/Develop/m/onnx-hipdnn-ep/mlir-integration  # Adjust to your path

# Set up directory variables
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build/test"
LOCAL_DIR="$PROJECT_ROOT/../../local"  # Two levels up from project root
TEMP_DIR="/c/temp"  # Windows Git Bash
# TEMP_DIR="/tmp"   # Linux
```

### Workflow Overview

```
1. Dump MLIR bytecode from E2E test
2. Convert bytecode to text format (save to TEMP_DIR)
3. Test complete pipeline to identify failing pass
4. Use debug flags to gather evidence (NOT speculation)
5. Isolate the failing pass
6. Identify root cause
7. Create LIT unit test for regression prevention
```

### Step 1: Enable Bytecode Dumping

Set `MORPHIZEN_DEBUG_MLIR_BACKEND=2` to dump MLIR bytecode:

```bash
MORPHIZEN_DEBUG_MLIR_BACKEND=2 \
XLNX_ONNX_EP_VERBOSE=2 \
"$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"
```

Find dump directory from output (look for `dump_dir:` line).

### Step 2: Convert Bytecode to Text

```bash
# Set dump directory from test output
DUMP_DIR="$TEMP_DIR/morphizen_dumps/<cache_key>"  # Replace <cache_key>

# Convert to text - SAVE TO TEMP DIRECTORY (NOT project workspace)
"$LOCAL_DIR/bin/mlir-opt" --allow-unregistered-dialect \
  "$DUMP_DIR/mlir_bytecode_dump.mlir" \
  -o "$TEMP_DIR/input.mlir"
```

**CRITICAL**: Save to `$TEMP_DIR/input.mlir`, NOT the project workspace.

### Step 3: Test Complete Pipeline

Use `hip-opt` with `--all-passes` to test the full ONNX→HIP→LLVM→Interface pipeline:

```bash
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  "$TEMP_DIR/input.mlir" \
  -o "$TEMP_DIR/output.mlir"
```

The `--all-passes` flag runs the complete compilation pipeline:
1. `--convert-onnx-to-hip` - Convert ONNX operations to HIP dialect
2. `--func.func='buffer-loop-hoisting,ownership-based-buffer-deallocation,optimize-allocation-liveness'` - Buffer management passes
3. `--canonicalize` - Canonicalization pass
4. `--hip-memory-pooling` - Memory pooling optimization
5. `--convert-hip-to-llvm` - Convert HIP dialect to LLVM dialect
6. `--hip-generate-interface` - Generate C interface wrapper functions

If successful, this validates that all MLIR transformations work correctly.

### Step 4: Gather Evidence with Debug Flags

**CRITICAL**: Don't speculate about what's wrong. Use MLIR's built-in debug flags to gather evidence.

#### Method A: Use --debug-only for Dialect Conversion Failures

When you see errors like "failed to legalize operation 'onnx.Conv'", the error message is intentionally cryptic. Use debug flags to see the actual reason:

```bash
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --debug-only=dialect-conversion \
  --mlir-disable-threading \
  "$TEMP_DIR/input.mlir" 2>&1 | tee "$TEMP_DIR/debug_conversion.log"
```

**What this reveals**:
- Pattern matching failures with specific reasons (e.g., "Conv operation missing required attributes")
- Which conversion pattern was attempted
- Why the pattern failed to match

**Example output**:
```
* Pattern : 'onnx.Conv -> (hip.conv)' {
  ** Failure : Conv operation missing required attributes: dilations
} -> FAILURE : pattern failed to match
```

This immediately tells you the root cause without speculation.

#### Method B: Dump IR at Each Pass

Use `--mlir-print-ir-tree-dir` to automatically dump IR before/after each pass:

```bash
mkdir -p "$TEMP_DIR/ir_dumps"

"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --mlir-print-ir-tree-dir="$TEMP_DIR/ir_dumps" \
  --mlir-print-ir-before-all \
  --mlir-disable-threading \
  "$TEMP_DIR/input.mlir" 2>&1 | tee "$TEMP_DIR/debug_all_passes.log"
```

**Advantages**:
- Automatic file creation with meaningful names
- No manual extraction needed
- See exact IR state before each pass

#### Available MLIR Debug Flags

**Dialect Conversion Debug Flags**:
- `--debug-only=dialect-conversion` - Show pattern matching attempts and failures with reasons
- `--debug-only=greedy-rewriter` - Show greedy pattern rewriter decisions

**IR Dumping Flags**:
- `--mlir-print-ir-before=<pass-name>` - Print IR before a specific pass
- `--mlir-print-ir-after=<pass-name>` - Print IR after a specific pass
- `--mlir-print-ir-before-all` - Print IR before all passes
- `--mlir-print-ir-after-all` - Print IR after all passes
- `--mlir-print-ir-tree-dir=<dir>` - Automatically dump IR to files (RECOMMENDED)
- `--mlir-print-ir-module-scope` - Always print at module scope (recommended for readability)

**Other Debug Flags**:
- `--mlir-disable-threading` - Disable multi-threading for deterministic output (REQUIRED for debugging)
- `--mlir-timing` - Display pass execution timing
- `--mlir-pass-statistics` - Display pass statistics

**Pass names** (use with `--mlir-print-ir-before=<name>` or `--mlir-print-ir-after=<name>`):
- `convert-onnx-to-hip` - ONNX to HIP conversion
- `buffer-loop-hoisting` - Buffer loop hoisting (nested pass)
- `ownership-based-buffer-deallocation` - Buffer deallocation (nested pass)
- `optimize-allocation-liveness` - Allocation liveness optimization (nested pass)
- `canonicalize` - Canonicalization
- `hip-memory-pooling` - Memory pooling
- `convert-hip-to-llvm` - HIP to LLVM conversion
- `hip-generate-interface` - Interface generation

### Step 5: Isolate the Failing Pass

Once you identify the failing pass, isolate it for focused debugging:

#### Method 1: Run Passes Incrementally

Run passes one-by-one up to the failing pass:

```bash
# Run only passes BEFORE the failing pass
# Example: If convert-hip-to-llvm fails, run up to hip-memory-pooling

"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-onnx-to-hip \
  --func.func='buffer-loop-hoisting,ownership-based-buffer-deallocation,optimize-allocation-liveness' \
  --canonicalize \
  --hip-memory-pooling \
  -o "$TEMP_DIR/input_before_failing_pass.mlir" \
  "$TEMP_DIR/input.mlir"

# Now test the failing pass in isolation
"$BUILD_DIR/bin/Debug/hip-opt" \
  --convert-hip-to-llvm \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  "$TEMP_DIR/input_before_failing_pass.mlir" 2>&1 | tee "$TEMP_DIR/isolated_pass_debug.log"
```

#### Method 2: Extract from IR Tree Dumps

Use the IR files created by `--mlir-print-ir-tree-dir`:

```bash
# Find the IR file dumped before the failing pass
IR_FILE=$(ls -t "$TEMP_DIR/ir_dumps"/pipeline_*before*.mlir | head -1)
echo "Input IR: $IR_FILE"

# Run failing pass in isolation
"$BUILD_DIR/bin/Debug/hip-opt" \
  --<failing-pass-name> \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  --mlir-disable-threading \
  "$IR_FILE" 2>&1 | tee "$TEMP_DIR/isolated_pass_debug.log"
```

### Step 6: Identify Root Cause

Analyze the debug output to identify the root cause:

```bash
# Search for pattern matching failures
grep "Pattern.*FAILURE" "$TEMP_DIR/debug_conversion.log"

# Search for errors
grep -E "(error|failed|mismatch)" "$TEMP_DIR/isolated_pass_debug.log"

# Review the IR at the failure point
cat "$TEMP_DIR/input_before_failing_pass.mlir"
```

### Complete Example: Debugging ONNX Conv Attribute Issue

Here's the complete workflow that was used to debug the "failed to legalize onnx.Conv" issue:

```bash
# Set up environment
cd /c/Develop/m/onnx-hipdnn-ep/mlir-integration
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build/test"
LOCAL_DIR="$PROJECT_ROOT/../../local"  # Two levels up from project root
TEMP_DIR="/c/temp"

# Run E2E test with bytecode dump
MORPHIZEN_DEBUG_MLIR_BACKEND=2 "$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"

# Convert bytecode (SAVE TO TEMP)
DUMP_DIR="$TEMP_DIR/morphizen_dumps/abc123"
"$LOCAL_DIR/bin/mlir-opt" --allow-unregistered-dialect \
  "$DUMP_DIR/mlir_bytecode_dump.mlir" -o "$TEMP_DIR/input.mlir"

# Test pipeline
"$BUILD_DIR/bin/Debug/hip-opt" --all-passes "$TEMP_DIR/input.mlir"
# ERROR: "failed to legalize operation 'onnx.Conv'"

# Use debug flags (NOT speculation)
"$BUILD_DIR/bin/Debug/hip-opt" \
  --all-passes \
  --debug-only=dialect-conversion \
  --mlir-disable-threading \
  "$TEMP_DIR/input.mlir" 2>&1 | tee "$TEMP_DIR/debug.log"

# Analyze evidence
grep "Pattern.*onnx.Conv" "$TEMP_DIR/debug.log"
# RESULT: "** Failure : Conv operation missing required attributes: dilations"

# ROOT CAUSE: Pattern expects dilations but ONNX spec says it's optional
# FIX: Update pattern to provide default [1,1]
```

### Best Practices for Evidence-Based Debugging

1. **Don't speculate - use debug flags first**
   - ❌ Bad: "Maybe it's a type mismatch" → spend hours guessing
   - ✅ Good: `--debug-only=dialect-conversion` → see exact failure in 30 seconds

2. **Save all temp files to temp directory, NOT project workspace**
   - ❌ Bad: `backend-mlir-compiler/test/input.mlir`
   - ✅ Good: `/c/temp/input.mlir`

3. **Work from project root - never use cd commands**
   - ❌ Bad: `cd backend-mlir-compiler/test && ../../build/...`
   - ✅ Good: `"$BUILD_DIR/bin/Debug/hip-opt" ...`

4. **Create LIT tests after identifying root cause**
   - See "Creating LIT Unit Tests from E2E Failures" section

### Common Issues

**Missing DLL dependencies (morphizen-mlir-compiler.dll)**:
- If the test fails to load `morphizen-mlir-compiler.dll` due to missing dependencies:
  ```bash
  # Workaround: Copy DLLs from local/bin to binary directory
  cp "$LOCAL_DIR/bin"/*.dll "$BUILD_DIR/Debug/bin/"
  ```

**Malformed null-terminated string error**:
- This occurred when `from_onnx_mlir = false` was hardcoded, causing all passes to be skipped
- Fixed by removing the `from_onnx_mlir` flag and always running the full pipeline

**Unknown dialect errors**:
- Ensure all required MLIR dialects are registered (ONNX, HIP, LLVM)
- Check that onnx-mlir is built correctly

**Pass failures**:
- Use `--debug-only=dialect-conversion` to see pattern matching failures with reasons
- Use `--mlir-print-ir-tree-dir` to automatically dump IR at each stage
- Extract the input IR and run the specific pass in isolation

**Pattern matching failures in ONNX→HIP conversion**:
- Use `--debug-only=dialect-conversion` to see why patterns failed
- Review the ONNX operations in the input IR
- Check if all required ONNX ops have HIP lowering patterns registered
- Verify that optional attributes have proper default values

## Creating LIT Unit Tests from E2E Failures

After identifying the root cause of an E2E test failure, create a focused LIT unit test to prevent regression.

### When to Create a LIT Test

Create a LIT test when you:
- Identify a specific conversion pattern failure
- Find a missing attribute default value
- Discover an unsupported operation variant
- Fix a type conversion bug

**Rule of thumb**: If the bug affected a specific MLIR operation or pass, create a LIT test for that operation/pass in isolation.

### Where to Put LIT Tests

```
3rd-party/morphizen/morphizen-mlir-compiler/test/lit/
├── Conversion/
│   ├── onnx-to-hip/    # ONNX → HIP lowering (most common for E2E failures)
│   └── hip-to-llvm/
├── Transforms/
└── Integration/
```

### LIT Test Structure

```mlir
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Brief description of what this test validates
//
// This test validates:
// - Specific behavior being tested
// - Edge case being covered
// - Bug that was fixed
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_name(%input: tensor<...>) -> tensor<...> {
    // CHECK-LABEL: func.func @test_name

    %output = "onnx.Operation"(%input) {
      attr1 = value1
      // attr2 intentionally omitted to test default handling
    } : (tensor<...>) -> tensor<...>

    // CHECK: hip.operation
    // CHECK-SAME: {attr1 = value1, attr2 = default_value}

    return %output : tensor<...>
  }
}
```

### Example: Conv Missing Dilations Attribute

File: `3rd-party/morphizen/morphizen-mlir-compiler/test/lit/Conversion/onnx-to-hip/test_conv_missing_dilations.mlir`

```mlir
// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_conv_missing_dilations(
    %input: tensor<1x3x224x224xf32>,
    %weights: tensor<64x3x7x7xf32>,
    %bias: tensor<64xf32>
  ) -> tensor<1x64x112x112xf32> {
    // CHECK-LABEL: func.func @test_conv_missing_dilations

    // ONNX Conv with dilations OMITTED (testing default behavior)
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3]
      // dilations intentionally omitted - should default to [1, 1]
    } : (...) -> tensor<1x64x112x112xf32>

    // Verify dilations=[1,1] is present in lowered hip.conv
    // CHECK: hip.conv
    // CHECK-SAME: {dilations = [1, 1],

    return %output : tensor<1x64x112x112xf32>
  }
}
```

### Running LIT Tests

```bash
# Via CTest
ctest --test-dir "$BUILD_DIR" -R LitTests --verbose

# Via llvm-lit
llvm-lit -v "$PROJECT_ROOT/3rd-party/morphizen/morphizen-mlir-compiler/test/lit/Conversion/onnx-to-hip/test_conv_missing_dilations.mlir"

# Manual verification during development
"$BUILD_DIR/bin/Debug/hip-opt" \
  "$PROJECT_ROOT/3rd-party/morphizen/.../test_conv_missing_dilations.mlir" \
  --convert-onnx-to-hip | FileCheck "$PROJECT_ROOT/3rd-party/morphizen/.../test_conv_missing_dilations.mlir"
```

### Workflow: From E2E Failure to LIT Test

```bash
# 1. Debug E2E failure (see Troubleshooting section)
# ... ROOT CAUSE IDENTIFIED: missing dilations attribute

# 2. Create minimal LIT test
# (Create file in appropriate test/lit/ subdirectory)

# 3. Verify test FAILS before fix (reproduces bug)
"$BUILD_DIR/bin/Debug/hip-opt" test.mlir --convert-onnx-to-hip | FileCheck test.mlir
# Expected: FAILS

# 4. Implement fix in conversion pattern

# 5. Verify test PASSES after fix
"$BUILD_DIR/bin/Debug/hip-opt" test.mlir --convert-onnx-to-hip | FileCheck test.mlir
# Expected: PASSES

# 6. Verify E2E test also passes
"$BUILD_DIR/Debug/bin/mlir_e2e_test.exe"

# 7. Clean up temp files (NOT in project workspace)
rm "$TEMP_DIR/input.mlir" "$TEMP_DIR/output.mlir"
rm -rf "$TEMP_DIR/ir_dumps"
```

### Best Practices for LIT Tests

1. **One test = One behavior** - Don't test multiple features in a single test
2. **Use descriptive names** - `test_conv_missing_dilations.mlir` ✅, `test_conv2.mlir` ❌
3. **Document the "why"** - Explain what bug this prevents
4. **Test edge cases** - Missing optional attributes, boundary values
5. **Keep tests minimal** - Small tensors, simple operations
6. **Use CHECK-LABEL** - Prevents false matches from other functions

### Reference Documentation

- **LIT Test Guide**: `3rd-party/morphizen/morphizen-mlir-compiler/test/lit/README.md`
- **FileCheck Syntax**: https://llvm.org/docs/CommandGuide/FileCheck.html
- **Existing Examples**: Browse `3rd-party/morphizen/.../test/lit/Conversion/onnx-to-hip/`

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

# These should NOT exist (workspace pollution - use /c/temp/ instead):
# ├── ir_dumps/                  # Should be in /c/temp/ir_dumps/
# └── two_layer_conv.mlir        # Should be in /c/temp/input.mlir
```

## Best Practices Summary

### Debugging Workflow
1. ✅ Work from project root - set `$PROJECT_ROOT`, `$BUILD_DIR`, `$LOCAL_DIR` variables
2. ✅ Save temp files to `/c/temp/` or `/tmp/` - NEVER in project workspace
3. ✅ Use debug flags FIRST (`--debug-only=dialect-conversion`) - gather evidence, not speculation
4. ✅ Use `--mlir-print-ir-tree-dir` for automatic IR dumps
5. ✅ Test isolated passes, not just full pipeline
6. ✅ Create LIT unit tests after identifying root cause

### Anti-Patterns to Avoid
1. ❌ NEVER use `cd` commands in workflows (except initial navigation to project root)
2. ❌ NEVER save MLIR files to project workspace (pollutes version control)
3. ❌ NEVER speculate about bugs - use debug flags to get evidence
4. ❌ NEVER skip creating LIT tests after fixing bugs (prevents regression)

### Path Examples
```bash
# ✅ Good - All commands from project root
cd /c/Develop/m/onnx-hipdnn-ep/mlir-integration
"$BUILD_DIR/bin/Debug/hip-opt" --all-passes "$TEMP_DIR/input.mlir"

# ❌ Bad - Using cd, relative paths, workspace pollution
cd backend-mlir-compiler/test
../../build/test/bin/Debug/hip-opt --all-passes two_layer_conv.mlir
```

### Debug Flag Priority
1. **First**: `--debug-only=dialect-conversion` - Pattern matching failures with reasons
2. **Second**: `--mlir-print-ir-tree-dir` - Automatic IR dumps
3. **Always**: `--mlir-disable-threading` - Deterministic debugging output
