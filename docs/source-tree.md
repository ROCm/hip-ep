<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Source tree and ownership

Public C++ declarations live under `include/hip`; implementations live under
the matching component in `lib`. Compiler C declarations use `include/hip-c`
and their implementation uses `lib/CAPI`, following MLIR and torch-mlir.

| Component | Current owner |
|---|---|
| HIP operation definitions, interfaces and shape rules | [Public IR](../include/hip/Dialect/Hip/IR) and [implementation](../lib/Dialect/Hip/IR) |
| HIP transforms and existing pipeline builders | [Public transforms](../include/hip/Dialect/Hip/Transforms) and [implementation](../lib/Dialect/Hip/Transforms) |
| Source-to-destination lowering | [Conversions](../lib/Conversion) |
| Compiler orchestration and plugin registry | [Compiler](../lib/Compiler) |
| Compiler C API | [Public headers](../include/hip-c) and [implementation](../lib/CAPI) |
| Host LLVM translation and artifact linking | [Target/LLVM](../lib/Target/LLVM) |
| Runtime wrappers and current device kernels | [Runtime](../lib/Runtime) |
| MorphiZen and ONNX Runtime adapter | [EP integration](../backend-mlir-compiler) |

The named HIP dialect directories and C API directories are the first completed
mechanical moves. Existing consumers can still include headers through
`hip/Dialect/IR`, `hip/Dialect/Transforms` and `hip/compiler_api.h`; these paths
forward to the canonical declarations. New code uses the named dialect paths
and `hip-c/compiler_api.h`. The C API target remains `HipCInterface`, and its
function names, types, symbol visibility and installed library name are unchanged.

TableGen output remains in the build include tree under the named dialect.
Keep generated-target dependencies explicit. Include generated fragments through
their canonical paths, including `hip/Dialect/Hip/Transforms/Passes.h.inc`.

Operation spellings, pass names and ordering are unchanged by these moves.
The [pipeline pass menu](pipeline_pass_menu.md) remains the ordering reference.
Build and installation commands remain in the [Linux](quick_start_linux.md)
and [Windows](quick_start.md) quick starts. Run the existing compiler suite with
`cmake --build <build> --target check-hip-mlir-lit`.

The [approved target layout](design/source-layout.md) describes the remaining
runtime, backend, integration and test moves and the later GGUF/Loom contributions.
Those proposed directories are not yet implemented. x4orward remains an external
artifact consumer.
