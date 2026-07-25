<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# LIT-based MLIR Tests

This directory contains LLVM Integrated Tester (LIT) tests for MLIR passes in the ONNX HIP Execution Provider.

## Directory Structure

```
test/lit/
├── Conversion/          # Dialect conversion passes
│   ├── onnx-to-hip/    # ONNX → HIP lowering tests
│   └── hip-to-llvm/    # HIP → LLVM lowering tests
├── Transforms/          # MLIR transformation passes
│   ├── BufferDeallocation/  # Automatic memory management tests
│   └── GenerateInterface/   # C-ABI wrapper generation tests
└── Integration/         # Multi-pass pipeline tests
```

## How LIT Config Works

LIT uses a two-file pattern. Understanding this matters when paths go wrong.

```
SOURCE tree                         BUILD tree
─────────────────────────────────   ──────────────────────────────────────
test/lit/lit.cfg.py   ← static      test/lit/lit.site.cfg.py  ← generated
test/lit/**/*.mlir    ← tests                by CMake from lit.site.cfg.py.in
```

**Load order (lit's perspective):**

1. You invoke `llvm-lit <build>/test/lit/` (always the build dir).
2. LIT searches upward from that path and auto-discovers `lit.site.cfg.py`
   in the build tree. It loads this **first**.
3. `lit.site.cfg.py` sets `config.llvm_tools_dir`, `config.hip_build_dir`,
   `config.hip_build_mode` — paths only CMake knows at configure time.
4. At the end, it calls `lit_config.load_config(config, ".../lit.cfg.py")`.
5. `lit.cfg.py` runs **second**. It defines the test suite (name, format,
   suffixes, tool substitutions). It can safely use the attributes set in step 3.

**Why you must point at the build dir:** `lit.site.cfg.py` only exists there.
Pointing at the source dir means LIT never finds the site config and
`config.llvm_tools_dir` is never set, causing `AttributeError` at startup.

## Running Tests

All `llvm-lit` commands must target the **build** directory. Run from the
project root (same directory as the top-level `CMakeLists.txt`).

```bash
BUILD=../build/$(basename $PWD)

# All LIT tests
llvm-lit -v $BUILD/test/lit/

# Specific subdirectory
llvm-lit -v $BUILD/test/lit/Conversion/onnx-to-hip/

# Single test — use --filter (regex on test path), not a source file path
llvm-lit -v $BUILD/test/lit/ --filter test_gemm_basic

# See full tool output (not just pass/fail summary)
llvm-lit -v --show-all $BUILD/test/lit/ --filter test_gemm_basic
```

### Via CTest

```bash
BUILD=../build/$(basename $PWD)

# LIT tests only
ctest --test-dir $BUILD -R MorphizenMLIRLitTests --verbose

# All tests (LIT + E2E)
ctest --test-dir $BUILD --verbose
```

### Via CMake Target

```bash
cmake --build ../build/$(basename $PWD) --target check-hip-mlir-lit
```

## Writing New Tests

### Test File Format

LIT tests use the following structure:

```mlir
// Brief description of what this test verifies
// RUN: hip-mlir-opt %s --pass-name | FileCheck %s

module {
  func.func @test_name(%arg: type) -> type {
    // CHECK-LABEL: func.func @test_name
    // CHECK: pattern to match in output

    %result = operation(%arg) : (type) -> type

    return %result : type
  }
}
```

### Key Components

1. **RUN line**: Specifies the command to run
   - `%s` = current file path
   - `hip-mlir-opt` = MLIR optimization tool
   - `--pass-name` = pass to test
   - `FileCheck %s` = verify output matches CHECK patterns

2. **CHECK directives**:
   - `CHECK-LABEL:` - Mark function/region boundaries
   - `CHECK:` - Match exact pattern
   - `CHECK-SAME:` - Continue previous CHECK on same line
   - `CHECK-NEXT:` - Must match immediately following line
   - `CHECK-NOT:` - Pattern must NOT appear
   - `CHECK-DAG:` - Match unordered patterns

3. **Pattern variables**:
   - `%[[VAR:.*]]` - Capture value to variable
   - `%{{.*}}` - Match any SSA value
   - `{{.*}}` - Match any text

### Example: Testing ONNX → HIP Conversion

```mlir
// Test ONNX ReLU → HIP ReLU lowering
// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_relu(%ctx: !hip.context, %input: memref<10xf32>) -> memref<10xf32> {
    // CHECK-LABEL: func.func @test_relu
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: memref<10xf32>

    %output = "onnx.Relu"(%input) : (memref<10xf32>) -> memref<10xf32>

    // CHECK: %[[OUT:.*]] = hip.relu(%[[CTX]], %[[INPUT]])
    // CHECK-SAME: : (!hip.context, memref<10xf32>) -> memref<10xf32>

    return %output : memref<10xf32>
  }
}
```

### Example: Testing Buffer Deallocation

```mlir
// Test automatic hip.free insertion
// RUN: hip-mlir-opt %s --bufferization-buffer-deallocation | FileCheck %s

module {
  func.func @test_alloc(%ctx: !hip.context) -> i32 {
    // CHECK-LABEL: func.func @test_alloc

    // CHECK: %[[BUF:.*]] = hip.alloc
    %buf = hip.alloc(%ctx) : memref<10xf32, 1>

    // CHECK: hip.free(%{{.*}}, %[[BUF]])

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
```

### Example: Testing Multi-Pass Pipeline

```mlir
// Test ONNX → HIP → LLVM pipeline
// RUN: hip-mlir-opt %s --convert-onnx-to-hip --convert-hip-to-llvm | FileCheck %s

module {
  func.func @pipeline(%ctx: !hip.context, %in: memref<10xf32>) {
    // CHECK-LABEL: llvm.func @pipeline

    %out = "onnx.Relu"(%in) : (memref<10xf32>) -> memref<10xf32>

    // Should see LLVM runtime call after full pipeline
    // CHECK: llvm.call @miopenActivationForward

    return
  }
}
```

## Test Categories

### Conversion Tests (`Conversion/`)

Test dialect conversion passes that lower operations from one dialect to another.

**onnx-to-hip/**:
- Conv lowering (basic, strided, grouped)
- GEMM lowering
- Activation operations (ReLU, etc.)
- Constant handling

**hip-to-llvm/**:
- Memory operations (alloc, free, handle lifecycle)
- Conv LLVM lowering
- GEMM LLVM lowering

### Transform Tests (`Transforms/`)

Test MLIR transformation passes that optimize or augment IR.

**BufferDeallocation/**:
- Automatic `hip.free` insertion
- Ownership tracking (function args vs local allocations)
- Multi-buffer management

**GenerateInterface/**:
- C-ABI wrapper generation for ONNX Runtime integration
- Pointer-to-memref conversion

### Integration Tests (`Integration/`)

Test complete compilation pipelines with multiple passes.

- ONNX → HIP → LLVM full pipeline
- Buffer management in multi-pass scenarios
- Complex operation sequences

## Common Passes

| Pass Name | Purpose | Example |
|-----------|---------|---------|
| `--convert-onnx-to-hip` | Lower ONNX ops to HIP dialect | `onnx.Conv` → `hip.conv` |
| `--convert-hip-to-llvm` | Lower HIP ops to LLVM calls | `hip.conv` → `llvm.call @miopenConv...` |
| `--bufferization-buffer-deallocation` | Insert automatic buffer cleanup | `hip.alloc` → `hip.alloc` + `hip.free` |
| `--generate-interface` | Generate C-ABI wrappers | `func.func @f` → `llvm.func @f_cabi` |

## Tips for Writing Tests

1. **Keep tests focused**: One test = one feature/behavior
2. **Use descriptive names**: `test_conv_stride2.mlir` not `test1.mlir`
3. **Add comments**: Explain what's being tested and why
4. **Test edge cases**: Not just happy paths
5. **Use CHECK-LABEL**: Prevents false matches across functions
6. **Capture variables**: Use `%[[VAR:.*]]` for readability
7. **Test negative cases**: Use CHECK-NOT for things that shouldn't happen

## Debugging Failed Tests

When a test fails:

```bash
# Run with verbose output
llvm-lit -v test/lit/path/to/test.mlir

# See the actual MLIR output
hip-mlir-opt test/lit/path/to/test.mlir --pass-name

# Compare expected vs actual
hip-mlir-opt test/lit/path/to/test.mlir --pass-name | FileCheck test/lit/path/to/test.mlir -v
```

## Prerequisites

For build prerequisites and setup, see [docs/quick_start.md](../../docs/quick_start.md).

Additional dependencies:
- **llvm-lit**: `pip install lit`

## References

- [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)
- [FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
- [MLIR Testing Guide](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)
