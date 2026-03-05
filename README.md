<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# hip Compiler

Standalone MLIR-based compiler infrastructure for AMD GPUs via HIP/MIOpen.

This is a portable subproject that provides:
1. **C API** (`hip-compiler.dll`) - DLL-safe interface using morphizen FileReader/FileWriter
2. **C++ Library** (`UdnaCompiler`) - Reusable CompilerDriver for MLIR → DLL/Object/IR compilation
3. **CLI Tools** (`hip-opt`, `hip-compile`) - Standalone compilation tools

## Architecture

```
hip-compiler/
├── include/hip-compiler/  # Public headers
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
│   ├── hip-opt/        # Pass runner (debugging)
│   ├── hip-compile/    # MLIR → DLL compiler
│   └── test-model-dll/  # E2E test runner for compiled DLLs
│
└── test/                     # LIT and E2E tests
```

## Building

### Prerequisites

- **LLVM/MLIR** (required) - with clang, LLD linker, and test utilities (FileCheck, lit)
- **ONNX-MLIR** (required) - ONNX dialect
- **Python 3** (required) - for lit tests
- **Python filecheck** (required) - `pip install filecheck` (used by LIT tests)
- **HIP/MIOpen/hipBLASLt** (optional) - for real runtime (otherwise mock runtime is used)

### Quick Start

```bash
# Configure (Mock Runtime - no GPU required)
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/hip-compiler \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  --fresh

# Build
cmake --build ../../build/hip-compiler --config Debug --parallel

# Test
ctest --test-dir ../../build/hip-compiler --verbose
```

### Build Options

- `BUILD_COMPILER_DLL` - Build hip-compiler.dll (default: ON, future work)
- `BUILD_STANDALONE_TOOLS` - Build hip-opt and hip-compile (default: ON)
- `BUILD_MOCK_RUNTIME` - Build with mock runtime (no GPU required) (default: ON)
- `ONNX_HIP_INCLUDE_LIT_TESTS` - Include LIT tests (default: ON)

### Real Runtime Build (GPU Required)

To compile models that run on an actual AMD GPU, you need the
[TheRock](https://github.com/ROCm/TheRock) ROCm SDK and must set two
environment variables **before** configuring CMake.

#### 1. Set environment variables

| Variable | Purpose | Example |
|---|---|---|
| `THEROCK_DIST` | Root of the TheRock SDK distribution | `D:\Develop\m\dist\therock` |
| `HIP_PLATFORM` | Target HIP platform | `amd` |

**Linux / macOS (bash)**
```bash
export THEROCK_DIST=/path/to/therock
export HIP_PLATFORM=amd
```

**Windows (cmd)**
```cmd
set "THEROCK_DIST=D:\Develop\m\dist\therock"
set HIP_PLATFORM=amd
```

> **Tip (Windows):** Quote the `set` command as shown to avoid trailing
> whitespace in the variable value, which will cause build failures.

#### 2. Configure with `-DBUILD_MOCK_RUNTIME=OFF`

```bash
LOCAL_DIR=$(cd ../../local && pwd)

cmake -S . -B ../../build/hip-compiler \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh
```

The build system will use Clang from `LLVM/MLIR` to compile the real HIP/MIOpen
runtime bitcode modules that get embedded into every generated DLL.

#### 3. Build

```bash
cmake --build ../../build/hip-compiler --config Debug --parallel
```

#### 4. Runtime environment for execution

When **running** a compiled DLL (either via `test-model-dll` or your own host
program), the HIP/MIOpen shared libraries must be discoverable. Set the
following environment variables at runtime:

| Variable | Purpose |
|---|---|
| `PATH` (Windows) / `LD_LIBRARY_PATH` (Linux) | Must include `$THEROCK_DIST/bin` (Windows) or `$THEROCK_DIST/lib` (Linux) so the OS can find `amdhip64.dll`, `MIOpen.dll`, etc. |
| `THEROCK_DIST` | Still required at **compile time** (when `hip-compile` generates the DLL) so that `lld-link` can locate the import libraries. Not needed at DLL execution time. |

**Example (Windows cmd):**
```cmd
set "THEROCK_DIST=D:\Develop\m\dist\therock"
set PATH=D:\Develop\m\dist\therock\bin;%PATH%

:: Compile the model
hip-compile model.mlir -o model.dll --mode dll -v

:: Run inference
test-model-dll model.dll --verbose --validate
```

**Example (Linux bash):**
```bash
export THEROCK_DIST=/path/to/therock
export LD_LIBRARY_PATH=$THEROCK_DIST/lib:$LD_LIBRARY_PATH

hip-compile model.mlir -o model.dll --mode dll -v
test-model-dll model.dll --verbose --validate
```

#### Linker notes

- `CompilerDriver` reads `THEROCK_DIST` at DLL-link time and adds
  `$THEROCK_DIST/lib` to the library search path.
- It links `amdhip64`, `MIOpen`, and `hipblaslt`. On Windows, if only a
  MinGW-style import library (`libhipblaslt.dll.a`) is available instead of a
  COFF `.lib`, the linker accepts it via a full-path fallback.

## Usage

### CLI Tool

```bash
# Compile MLIR to DLL
hip-compile input.mlir -o output.dll --from-onnx-mlir -O2 -v

# Debug MLIR passes
hip-opt input.mlir --morphizen-pipeline
hip-opt input.mlir --convert-onnx-to-hip --convert-hip-to-llvm
```

### C++ Library

```cpp
#include "hip-compiler/Compiler/CompilerDriver.h"

using namespace hip::compiler;

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
#include <hip-compiler/compiler_api.h>

CompilationOptions opts;
morphizen_mlir_get_default_options(&opts);
opts.from_onnx_mlir = 1;
opts.opt_level = 2;

CompilerError error;
CompilerErrorCode result = hip_compile(
    &input_reader, &output_writer, &opts, &error
);
```

## Documentation

See `docs/` directory for detailed documentation:

- **Design** (`docs/design/`) - Architecture, dialects, passes, runtime
- **Guides** (`docs/guides/`) - Demos, GPU testing, troubleshooting
- **Integration** (`docs/integration/`) - ONNX-MLIR integration, API usage

## Dependencies

### Required
- LLVM/MLIR (compiler infrastructure)
- ONNX-MLIR (ONNX dialect)
- LLD linker (DLL linking)
- Protobuf (metadata serialization)

### Optional
- HIP/MIOpen/hipBLASLt (real runtime - not needed for mock)

### NOT Dependencies
- ❌ authentication module (used by level-1-pass, but not this subproject)
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
