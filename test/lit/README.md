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
│   ├── onnx-to-hip/     # ONNX → HIP lowering tests
│   ├── onnx-to-hipdnn/  # ONNX → hipDNN lowering tests (opt-in path)
│   ├── onnx-to-hipsr/   # ONNX → HIPSR lowering tests (opt-in path)
│   ├── hip-to-llvm/     # HIP → LLVM lowering tests
│   ├── hip-to-tosa/     # HIP → TOSA lowering tests
│   └── hipsr-to-llvm/   # HIPSR → LLVM lowering tests (opt-in path)
├── Dialect/             # Dialect verifiers, folders, shape inference
├── Transforms/          # MLIR transformation passes
│   ├── AssignOpStateSlots/   # Op-state slot assignment
│   ├── GenerateInterface/    # C-ABI wrapper generation
│   ├── GenerateOpStateInit/  # Op-state init function generation
│   ├── HipFusionTransform/   # HIP op fusion
│   └── ShareMatMulNBitsActivation/ # MatMulNBits activation-sharing groups
├── Pipeline/            # Multi-pass pipeline tests
├── Plugin/              # Compiler plugin tests
├── e2e/                 # Models also compiled and run by the E2E suite
└── Inputs/              # Shared inputs for split/included tests
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
llvm-lit -v $BUILD/test/lit/ --filter onnx-to-hip/test_gemm

# See full tool output (not just pass/fail summary)
llvm-lit -v --show-all $BUILD/test/lit/ --filter onnx-to-hip/test_gemm
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

Abridged from `Conversion/onnx-to-hip/test_relu.mlir`:

```mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @relu_static_f32(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %y = "onnx.Relu"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @relu_static_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>)
  // CHECK-NOT: onnx.Relu
  // CHECK: hip.max(%[[CTX]]) ins(%[[X]]
}
```

### Example: Testing Multi-Pass Pipelines

Chain passes on the RUN line, or use a registered pipeline such as
`--onnx-to-hip-pipeline` or `--hip-to-llvm-pipeline`. See `Pipeline/` and
`Transforms/GenerateInterface/` for real examples, e.g.:

```mlir
// RUN: hip-mlir-opt --hip-pool-allocs --convert-hip-to-llvm %s | FileCheck %s
```

## Test Categories

### Conversion Tests (`Conversion/`)

Test dialect conversion passes that lower operations from one dialect to another.

- **onnx-to-hip/**: ONNX op lowering into the HIP dialect (default path).
- **onnx-to-hipdnn/**, **onnx-to-hipsr/**: the opt-in hipDNN (used when a
  hipDNN handle is available) and HIPSR (`HIPDNN_EP_HIPSR=1`) paths.
- **hip-to-llvm/**, **hipsr-to-llvm/**: lowering to the LLVM dialect
  (HIP ops become `llvm.call @wrap_*` runtime calls).
- **hip-to-tosa/**: HIP → TOSA lowering.

### Transform Tests (`Transforms/`)

Test MLIR transformation passes that optimize or augment IR.

- **AssignOpStateSlots/**, **GenerateOpStateInit/**: per-op runtime state
  slots and their init function.
- **GenerateInterface/**: C-ABI wrapper generation for ONNX Runtime
  integration.
- **HipFusionTransform/**: HIP op fusion.
- **ShareMatMulNBitsActivation/**: grouping of `hip.matmul_nbits` ops that
  share one quantized activation, including producer-led groups.

### Pipeline and Plugin Tests (`Pipeline/`, `Plugin/`)

Multi-pass pipelines (constant externalization, alloc pooling, module
metadata, pipeline overrides) and compiler plugin hooks.

## Common Passes

| Pass Name | Purpose | Example |
|-----------|---------|---------|
| `--convert-onnx-to-hip` | Lower ONNX ops to HIP dialect | `onnx.Conv` → `hip.conv` |
| `--convert-hip-to-llvm` | Lower HIP ops to LLVM calls | `hip.conv` → `llvm.call @wrap_conv` |
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
