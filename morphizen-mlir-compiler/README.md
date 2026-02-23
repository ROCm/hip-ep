<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Morphizen MLIR Compiler

Standalone MLIR-based compiler infrastructure for AMD GPUs via HIP/MIOpen.

This is a portable subproject that provides:
1. **C API** (`morphizen-mlir-compiler.dll`) - DLL-safe interface using morphizen FileReader/FileWriter
2. **C++ Library** (`MorphizenCompiler`) - Reusable CompilerDriver for MLIR → DLL/Object/IR compilation
3. **CLI Tools** (`morphizen-opt`, `morphizen-compile`) - Standalone compilation tools

## Architecture

```
morphizen-mlir-compiler/
├── include/morphizen-mlir-compiler/  # Public headers
│   ├── Dialect/Hip/          # HIP dialect (IR, types, ops)
│   │   ├── IR/               # Dialect definition
│   │   └── Transforms/       # HIP→HIP optimizations
│   ├── Conversion/           # Cross-dialect conversions
│   │   ├── OnnxToHip/        # ONNX→HIP lowering
│   │   ├── HipToLLVM/        # HIP→LLVM lowering
│   │   └── Passes.h          # Umbrella header
│   ├── Compiler/             # Application-level
│   │   ├── Passes/           # App-specific passes
│   │   ├── Pipeline.h        # SINGLE SOURCE OF TRUTH
│   │   └── CompilerDriver.h  # End-to-end driver
│   ├── Target/LLVM/          # Code generation
│   └── InitAllPasses.h       # Convenience header
│
├── lib/                      # Implementation (mirrors include/)
│   ├── Dialect/Hip/
│   │   ├── IR/               # HipDialect, ops, types
│   │   └── Transforms/       # MemoryPooling, BufferDeallocation
│   ├── Conversion/
│   │   ├── OnnxToHip/        # ONNX→HIP conversion
│   │   ├── HipToLLVM/        # HIP→LLVM conversion
│   │   └── Passes.cpp        # Registration
│   ├── Compiler/
│   │   ├── Passes/           # GenerateInterface
│   │   ├── Pipeline/         # Pipeline definition
│   │   └── CompilerDriver.cpp
│   ├── Target/LLVM/          # LLVMBackend, DLLLinker
│   ├── Runtime/              # HIP/MIOpen wrappers
│   └── CInterface/           # C API wrapper
│
├── tools/
│   ├── morphizen-opt/        # Pass runner (debugging)
│   └── morphizen-compile/    # MLIR → DLL compiler
│
└── test/                     # LIT and E2E tests
```

## Building

### Prerequisites

- **LLVM/MLIR** (required) - with clang, LLD linker, and test utilities (FileCheck, lit)
- **ONNX-MLIR** (required) - ONNX dialect
- **Python 3** (required) - for lit tests
- **HIP/MIOpen/hipBLASLt** (optional) - for real runtime (otherwise mock runtime is used)

### Quick Start

```bash
# Configure (Mock Runtime - no GPU required)
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/morphizen-mlir-compiler \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  --fresh

# Build
cmake --build ../../build/morphizen-mlir-compiler --config Debug --parallel

# Test
ctest --test-dir ../../build/morphizen-mlir-compiler --verbose
```

### Build Options

- `BUILD_COMPILER_DLL` - Build morphizen-mlir-compiler.dll (default: ON, future work)
- `BUILD_STANDALONE_TOOLS` - Build morphizen-opt and morphizen-compile (default: ON)
- `BUILD_MOCK_RUNTIME` - Build with mock runtime (no GPU required) (default: ON)
- `ONNX_HIP_INCLUDE_LIT_TESTS` - Include LIT tests (default: ON)

### Real Runtime Build

Requires TheRock ROCm SDK:

```bash
export THEROCK_DIST=C:/Develop/m/dist/therock
export HIP_PLATFORM=amd
LOCAL_DIR=$(cd ../../local && pwd)

cmake -S . -B ../../build/morphizen-mlir-compiler \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh
```

## Usage

### CLI Tool

```bash
# Compile MLIR to DLL
morphizen-compile input.mlir -o output.dll --from-onnx-mlir -O2 -v

# Debug MLIR passes
morphizen-opt input.mlir --morphizen-pipeline
morphizen-opt input.mlir --convert-onnx-to-hip --convert-hip-to-llvm
```

### C++ Library

```cpp
#include "morphizen-mlir-compiler/Compiler/CompilerDriver.h"

using namespace morphizen::mlir_compiler;

CompilerDriver driver;
CompilationOptions opts;
opts.from_onnx_mlir = true;
opts.opt_level = 2;
opts.verbose = true;

std::string errorMessage;
bool success = pipeline.compile(inputMLIR, "output.dll", opts, errorMessage);
```

### C API (Future)

```c
#include <morphizen-mlir-compiler/compiler_api.h>

CompilationOptions opts;
morphizen_mlir_get_default_options(&opts);
opts.from_onnx_mlir = 1;
opts.opt_level = 2;

CompilerError error;
CompilerErrorCode result = morphizen_mlir_compile(
    &input_reader, &output_writer, &opts, &error
);
```

## Documentation

See `doc/` directory for detailed documentation:

- **Design** (`doc/design/`) - Architecture, dialects, passes, runtime
- **Guides** (`doc/guides/`) - Demos, GPU testing, troubleshooting
- **Integration** (`doc/integration/`) - ONNX-MLIR integration, API usage

## Dependencies

### Required
- LLVM/MLIR (compiler infrastructure)
- ONNX-MLIR (ONNX dialect)
- LLD linker (DLL linking)
- Protobuf (metadata serialization)

### Optional
- HIP/MIOpen/hipBLASLt (real runtime - not needed for mock)

### NOT Dependencies
- ❌ morphizen-core (used by level-1-pass, but not this subproject)
- ❌ morphizen-graph/pattern (level-1-pass only)
- ❌ glog (level-1-pass only)

This ensures the subproject is **portable** and **standalone**.

## Migration Readiness

This subproject is designed for eventual migration to the morphizen repository:

- ✅ No git submodules
- ✅ No dependencies on main repo (level-1-pass, custom-op-mlir)
- ✅ Self-contained build system
- ✅ AMD copyright headers on all files
- ✅ Standalone documentation

## License

Licensed under the MIT License. Copyright (C) 2026 Advanced Micro Devices, Inc.
