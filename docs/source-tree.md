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
| HIP transforms | [Pass declarations](../include/hip/Dialect/Hip/Transforms) and [implementations](../lib/Dialect/Hip/Transforms) |
| Source-to-destination lowering | [Conversions](../lib/Conversion) |
| HIP pipeline builders, compiler driver and plugin registry | [Pipeline API](../include/hip/Compiler/Pipelines.h) and [compiler implementation](../lib/Compiler) |
| Compiler C API | [Public headers](../include/hip-c) and [implementation](../lib/CAPI) |
| Host LLVM translation and artifact linking | [Target/LLVM](../lib/Target/LLVM) |
| Runtime wrappers and current device kernels | [Runtime](../lib/Runtime) |
| MorphiZen and ONNX Runtime adapter | [EP integration](../backend-mlir-compiler) |

Link `HipDialectTransforms` to use individual HIP transform factories without
the ONNX conversion libraries. Its rocMLIR fusion passes use `HipToTosa` to
check which operations can be lowered. Link `LibHipCompiler` for the HIP
pipeline builders, which compose conversions, transforms and plugin passes.
The existing `registerHipPasses()` also registers conversions; callers using
that aggregate need both `HipDialectTransforms` and `HipConversion`, or
`LibHipCompiler` which provides both.

Use `hip/Dialect/Hip/{IR,Transforms}` and `hip-c/compiler_api.h` for new code.
The compiler C API target remains `HipCInterface`; its symbols, types and
installed library name are unchanged.

For a source build, put both `<source>/include` and `<build>/include` on the
include path. CMake provides the old `hip/Dialect/{IR,Transforms}` and
`hip/compiler_*.h` paths under `<build>/include` and installs the old header
names alongside the new ones. Both `hip/Dialect/Transforms/Pipelines.h` and
`hip/Dialect/Hip/Transforms/Pipelines.h` forward to `hip/Compiler/Pipelines.h`
in the build and install trees. A tool that opens an old path directly in the
source tree must use the canonical path or the configured build path.

For TableGen, the old and new interface `.td` paths produce the same
declarations and definitions, including when used as the main input to
`mlir-tblgen`. The old `.inc` paths forward to the generated fragments and
support repeated inclusion with different macros. Depend on the corresponding
TableGen targets. The `.td` files and generated fragments are for build-tree
consumers and are not installed.

Operation spellings, pass names and ordering are unchanged by these moves.
The [pipeline pass menu](pipeline_pass_menu.md) remains the ordering reference.
Build and installation commands remain in the [Linux](quick_start_linux.md)
and [Windows](quick_start.md) quick starts. Run the existing compiler suite with
`cmake --build <build> --target check-hip-mlir-lit`.
