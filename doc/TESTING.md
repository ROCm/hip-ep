# Testing Guide for morphizen-mlir

## Overview

This document describes how to run and verify the MLIR integration tests for the morphizen-mlir project.

## Running Tests

### Quick Start

Simply run the test script:

```bash
cd d:\Develop\m\morphizen-mlir\tools
.\run_ort_integration_test.bat
```

This will:
1. Set the required environment variable (`MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend`)
2. Check for test executable and models
3. Generate test models if needed
4. Run all integration tests

### Manual Test Execution

If you want to run tests manually:

```bash
# Set environment variable to activate MLIR backend
set MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend

# Navigate to test directory
cd d:\Develop\m\morphizen-mlir\test

# Run the test executable
D:\Develop\m\build\morphizen-mlir\bin\ort_integration_test.exe
```

## Test Cases

The test suite includes 3 test cases:

### 1. LoadVitisAIProvider
Verifies that the VitisAI Execution Provider can be loaded and registered successfully.

### 2. CreateSessionWithVitisAIProvider (Conv Model)
Tests session creation with a simple Conv operation model.

**Model:** `conv_model.onnx`
- Input: X [1, 3, 8, 8]
- Operation: Conv (kernel=3x3, padding=1, stride=1)
- Output: Y [1, 16, 8, 8]

### 3. CreateSessionWithConvGemmModel
Tests session creation with a Conv+Flatten+Gemm pipeline model.

**Model:** `conv_gemm_model.onnx`
- Input: X [1, 3, 8, 8]
- Operations: Conv → Flatten → Gemm
- Output: Y [1, 32]

## Expected Output

### ModuleOp Content (from line 76 of pass_main.cpp)

When the MLIR pass executes, it prints the complete ModuleOp to stdout. Here's an example output from the Conv model test:

```mlir
ModuleOp content:
#loc2 = loc("graph_for_mlir.txt":2:25)
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{onnx.name = "X"}], function_type = (tensor<1x3x8x8xf32>) -> tensor<1x16x8x8xf32>, res_attrs = [{onnx.name = "Y"}], sym_name = "main_graph"}> ({
  ^bb0(%arg0: tensor<1x3x8x8xf32> loc("graph_for_mlir.txt":2:25)):
    // %arg0 is used by %4
    %2 = "onnx.None"() : () -> none loc(#loc3) // unused
    %3 = "arith.constant"() <{value = dense<"0x3D7ED6BC99AB30BD..."> : tensor<16x3x3x3xf32>}> {node.outputs = ["W"]} : () -> tensor<16x3x3x3xf32> loc(#loc4) // user: %4
    %4 = "onnx.Conv"(%arg0, %3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], node.outputs = ["Y"], onnx_node_name = "conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x8x8xf32> loc(#loc5) // user: %5
    "func.return"(%4) : (tensor<1x16x8x8xf32>) -> () loc(#loc6) // id: %5
  }) {onnx.graph.name = "resent50_by_vaip"} : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("graph_for_mlir.txt":1:1)
#loc1 = loc("graph_for_mlir.txt":2:3)
#loc3 = loc("graph_for_mlir.txt":3:10)
#loc4 = loc("graph_for_mlir.txt":4:12)
#loc5 = loc("graph_for_mlir.txt":5:10)
#loc6 = loc("graph_for_mlir.txt":6:5)
```

### Conv+Gemm Model Output

For the Conv+Gemm model, the ModuleOp shows a more complex pipeline:

```mlir
ModuleOp content:
#loc2 = loc("graph_for_mlir.txt":2:25)
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{onnx.name = "X"}], function_type = (tensor<1x3x8x8xf32>) -> tensor<1x32xf32>, res_attrs = [{onnx.name = "Y"}], sym_name = "main_graph"}> ({
  ^bb0(%arg0: tensor<1x3x8x8xf32> loc("graph_for_mlir.txt":2:25)):
    // Conv operation
    %2 = "onnx.None"() : () -> none
    %3 = "arith.constant"() <{value = dense<...> : tensor<16x3x3x3xf32>}> : () -> tensor<16x3x3x3xf32>
    %4 = "onnx.Conv"(%arg0, %3) {...} : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x8x8xf32>
    
    // Flatten operation
    %5 = "onnx.Flatten"(%4) {axis = 1 : si64} : (tensor<1x16x8x8xf32>) -> tensor<1x1024xf32>
    
    // Gemm operation
    %6 = "arith.constant"() <{value = dense<...> : tensor<1024x32xf32>}> : () -> tensor<1024x32xf32>
    %7 = "arith.constant"() <{value = dense<...> : tensor<32xf32>}> : () -> tensor<32xf32>
    %8 = "onnx.Gemm"(%5, %6, %7) {...} : (tensor<1x1024xf32>, tensor<1024x32xf32>, tensor<32xf32>) -> tensor<1x32xf32>
    
    "func.return"(%8) : (tensor<1x32xf32>) -> ()
  })
})
```

## Verification

### Success Indicators

A successful test run should show:

1. **VitisAI EP Registration:**
   ```
   [SetUp] VitisAI EP registered successfully from: onnxruntime_vitisai_ep.dll
   ```

2. **Session Creation:**
   ```
   [Test] Session created successfully with VitisAI EP!
   [Test] MLIR pass was executed during session creation
   ```

3. **MLIR Parsing:**
   ```
   Successfully parsed MLIR string to ModuleOp
   ModuleOp created, ready for MLIR transformations
   ```

4. **Operation Walking:**
   ```
   Walking operations in module...
   Total operations in module: 6  (for Conv model)
   Total operations in module: 10 (for Conv+Gemm model)
   ```

5. **Test Results:**
   ```
   [==========] Running 3 tests from 1 test suite.
   [  PASSED  ] 3 tests.
   ```

## Debug Mode

For detailed debug output, set additional environment variables before running tests:

```bash
set MORPHIZEN_DEBUG_MLIR=2
set MORPHIZEN_DEBUG_MLIR_GRAPH=2
set GLOG_logtostderr=1
set GLOG_minloglevel=0
set MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend
```

Then run the test executable directly to see all debug output.

## Generating Test Models

### Conv Model

```bash
cd d:\Develop\m\morphizen-mlir\test
python gen_conv_model.py
```

This generates `conv_model.onnx` with a simple Conv operation.

### Conv+Gemm Model

```bash
cd d:\Develop\m\morphizen-mlir\test
python gen_conv_gemm_model.py
```

This generates `conv_gemm_model.onnx` with Conv → Flatten → Gemm pipeline.

## Troubleshooting

### Test Executable Not Found

If you see "Test executable not found", build the project first:

```bash
cd d:\Develop\m\morphizen-mlir
.\build.bat
```

### Model Not Found

The test script will automatically generate missing models. If generation fails, run the Python scripts manually as shown above.

### MLIR Parsing Fails

If MLIR parsing fails, ensure:
1. `MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend` is set
2. The MLIR backend (mlir-imp) is properly built and linked
3. All required MLIR dialects are loaded (func, arith)

## Test Output Files

During test execution, the following files are generated in the test directory:

- `graph_for_mlir.txt` - MLIR IR representation of the graph
- `vaip_init/init-graph-for-mlir.onnx` - Initial graph saved by init pass

These are temporary files and should not be committed to git.

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Build
cd d:\Develop\m\morphizen-mlir
.\build.bat

# Test
cd tools
.\run_ort_integration_test.bat
```

The test script returns exit code 0 on success, non-zero on failure.
