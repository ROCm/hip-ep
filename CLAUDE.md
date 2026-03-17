<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

**UDNA Compiler**: Standalone MLIR-based compiler infrastructure for AMD GPUs. Provides C API, C++ library, and CLI tools for ONNX→MLIR→DLL compilation.

**Architecture**: 4 layers - Dialect (HIP) → Conversion (ONNX→HIP, HIP→LLVM) → Compiler (Pipeline, Passes, Driver) → Target (LLVM backend, DLL linker)

## Build System

**CRITICAL**: Work from project root directory.

**Build paths**:
- Build: \`../build/$(basename $PWD)\`
- Install: \`../../local\`
- Runtime: \`/MTd\` via \`CMAKE_MSVC_RUNTIME_LIBRARY\`

**Configure**:
\`\`\`bash
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON --fresh
\`\`\`

**Build**: \`cmake --build ../build/$(basename $PWD) --config Debug --parallel\`

**Test**: \`ctest --test-dir ../build/$(basename $PWD) --verbose\`

**Key Options**:
- \`BUILD_STANDALONE_TOOLS=ON\` (default) - Build hip-mlir-opt and hip-compiler
- \`BUILD_MOCK_RUNTIME=ON\` (default) - Build with mock runtime (no GPU required)
- \`ONNX_HIP_INCLUDE_LIT_TESTS=ON\` (default) - Include LIT tests

### Prebuilt Flow (fast, Release build)

See `docs/BUILDING-PREBUILT.md` for full instructions.

**One-time setup**: `bash scripts/setup-prebuilt.sh`

**Configure**:
\`\`\`bash
PREBUILT_DIR=$(cd ../prebuilt-local && pwd)
cmake -S . -B ../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$PREBUILT_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_MOCK_RUNTIME=ON -DONNX_HIP_INCLUDE_LIT_TESTS=ON --fresh
\`\`\`

**Build**: \`cmake --build ../build/$(basename $PWD) --config Release --parallel\`
**Test**: \`ctest --test-dir ../build/$(basename $PWD) --config Release --verbose\`

## Architecture

**Layers**: Applications (CLI tools, C API) → Compiler (Driver, Pipeline, Passes) → Conversion (ONNX→HIP, HIP→LLVM) → Dialect (HIP IR/Transforms) → Target (LLVM backend)

**Key Components**:
1. **HIP Dialect**: Custom MLIR dialect for HIP operations (lib/Dialect/Hip/)
2. **Conversion Passes**: ONNX→HIP, HIP→LLVM lowering (lib/Conversion/)
3. **Compiler Pipeline**: Orchestrates pass execution (lib/Compiler/Pipeline/)
4. **Compiler Driver**: End-to-end MLIR→DLL compilation (lib/Compiler/CompilerDriver.cpp)
5. **Target Backend**: LLVM IR optimization and DLL linking (lib/Target/LLVM/)

**Key Headers**: \`CompilerDriver.h\`, \`Pipeline.h\`, \`Passes.h\`, \`HipDialect.h\`, \`compiler_api.h\`

**Directories**: \`include/hip/\`, \`lib/\`, \`tools/hip-mlir-opt/\`, \`tools/hip-compiler/\`, \`dll/\`, \`test/\`, \`proto/\`

## Testing

**Framework**: LLVM LIT tests + CTest
**Run**: \`ctest --test-dir ../build/$(basename $PWD) --verbose\`

## Git Workflow

**Remotes**: \`origin\` (main repo) | \`fork\` (your fork, push here)

**CRITICAL**: Always push to \`fork\`, never \`origin\`

**Push Policy**: After creating commits, ALWAYS push to fork immediately unless the user says otherwise.

**Auto-PR Policy**: After successfully pushing to fork, IMMEDIATELY check if a PR exists for the branch. If not, create a draft PR automatically with \`gh pr create --draft\`.

**Required Steps**:
1. Sync: \`git checkout main && git pull origin main\`
2. Branch: \`git checkout -b feature/<name>\` (BEFORE changes)
3. Commit: After file changes, BEFORE testing
4. First Push: \`git push -u fork <branch>\` (sets upstream tracking)
5. Subsequent Pushes: \`git push fork <branch>\`
6. PR: \`gh pr create --draft\` (IMMEDIATELY after first push)

**CRITICAL - Before Marking PR Ready**:
\`\`\`bash
pre-commit run --all-files
\`\`\`
If pre-commit makes changes, commit and push BEFORE marking PR ready.

**Commit Rules**:
- ❌ NO AI mentions
- ✅ Conventional commits (\`feat:\`, \`fix:\`, \`docs:\`, etc.)
- ✅ Stage specific files: \`git add <file>\`

## Setup

**Pre-commit** (if available): \`scripts/setup-dev-env.ps1\` (Windows) or \`scripts/setup-dev-env.sh\` (Linux/Mac)

## Common Pitfalls

1. Build dir: \`../build/$(basename $PWD)\`, NOT \`./build\`
2. Install prefix: \`../../local\`
3. Git push: \`fork\`, NEVER \`origin\`
4. Never work on \`main\` branch
5. Launch bash from MSVC Developer Command Prompt (Windows)
6. NO AI/tool mentions in commits/PRs

## Docs

See \`README.md\`, \`docs/\`, \`test/README.md\`

## Dependencies

**Required**: LLVM/MLIR, ONNX-MLIR, LLD, FlatBuffers
**Optional**: HIP/MIOpen/hipBLASLt (for real runtime, otherwise mock)
