# CLAUDE.md - Project Instructions for ONNX HipDNN EP

## Critical Bazel Build Rules

### Always Use Remote Build
**IMPORTANT**: All Bazel builds in this project MUST use `--config=remote`:

```bash
bazel build //target:name --config=remote
bazel test //target:name --config=remote
```

Remote builds are significantly faster than local builds.

### Use bazel_dep for Dependencies
**Prefer bazel_dep over other dependency mechanisms**. Create a module in bazel-registry if needed.

### Always Use bazel-cpp-toolchain
**MANDATORY**: All C++ builds must use bazel-cpp-toolchain (clang-cl on Windows).

- Never use default MSVC cl.exe toolchain
- bazel-cpp-toolchain provides consistent, reproducible builds across platforms
- Ensures compatibility with LLVM/MLIR compilation

### NEVER Use local_path_override
**CRITICAL**: NEVER use `local_path_override` in MODULE.bazel files.

Instead, use `--override_module` in `.bazelrc.registries` for local development:

```bash
# .bazelrc.registries (gitignored)
common --override_module=bazel_cpp_toolchain=../../bazel-cpp-toolchain/bazel-cpp-toolchain-1
common --override_module=morphizen=../../MorphiZen-1
common --override_module=llvm-project-overlay=../../bazel-registry-1/modules/llvm-project-overlay/22.1.3
```

**Why?**
- `local_path_override` only works when that module is the root
- `local_path_override` is ignored when the module is a dependency
- `--override_module` works globally for all builds
- `.bazelrc.registries` keeps local paths out of version control

## Project Structure

### Backend MLIR Compiler
The `backend-mlir-compiler/` directory contains the MLIR-based compilation backend:
- `proto/` - Protobuf metadata definitions
- `level-1-pass/` - Compile-time MLIR compilation pass (uses `alwayslink=True`)
- `custom-op-mlir/` - Runtime custom op execution (uses `alwayslink=True`)
- `common/` - Shared utilities

## Dependencies

### Production Setup
- **MorphiZen**: Via bazel-registry module system
- **Bazel Registry**: https://github.com/ROCm/bazel-registry (local clone at bazel-registry-1)
- **Bazel C++ Toolchain**: From ROCm bazel-registry (clang-cl on Windows)

### Local Development Setup
All local overrides are configured in `.bazelrc.registries` (gitignored):

```bash
# Local registry for offline/faster access
common --registry=file:///C:/Develop/m/source/bazel-registry-1

# Module overrides for local development
common --override_module=bazel_cpp_toolchain=../../bazel-cpp-toolchain/bazel-cpp-toolchain-1
common --override_module=morphizen=../../MorphiZen-1
common --override_module=llvm-project-overlay=../../bazel-registry-1/modules/llvm-project-overlay/22.1.3
```

### Required Dependencies
- **Protobuf**: v33.4
- **glog**: v0.7.1
- **rules_proto**: v7.1.0
- **rules_python**: v1.7.0
- **LLVM/MLIR**: via llvm-project-overlay v22.1.3

## Known Issues (Fixed Locally)

### MorphiZen Issues

#### 1. Encryption Dependency
The MorphiZen `morphizen-core/BUILD.bazel` had an incorrect dependency:
- **Issue**: Line 386 referenced `"//encryption"` which doesn't exist
- **Fix**: Changed to `"//morphizen-foundation"` (encryption code is there)

#### 2. Missing cmake/scripts BUILD
- **Issue**: `cmake/scripts/xxd.py` referenced but no BUILD.bazel
- **Fix**: Created `cmake/scripts/BUILD.bazel` with `exports_files(["xxd.py"])`

### LLVM Project Overlay Issues

#### 1. zlib Include Path Issue (clang-cl)
Fixed in bazel-cpp-toolchain by changing quote_include_paths to use /I instead of -iquote.

## Build Verification Commands

```bash
# Build proto
bazel build //backend-mlir-compiler/proto:mlir_compilation_proto --config=remote

# Build level-1-pass
bazel build //backend-mlir-compiler/level-1-pass:morphizen-level1-pass-mlir-compiler --config=remote

# Build custom-op-mlir
bazel build //backend-mlir-compiler/custom-op-mlir:morphizen-custom-op-mlir --config=remote

# Build all backend-mlir-compiler targets
bazel build //backend-mlir-compiler/... --config=remote
```

## Coding Standards

- Use C++17 standard
- Follow existing code patterns in the codebase
- Use `alwayslink = True` for targets with static registration
- Header-only libraries use `cc_library` with only `hdrs` and `includes`

## Migration Path

1. **Current State**: Local path overrides for development
2. **Next Steps**:
   - Add MorphiZen as git submodule in `3rd-party/morphizen`
   - Configure ROCm bazel-registry for production builds
   - Remove local_path_override when ready for CI/CD
