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

## Project Structure

### Backend MLIR Compiler
The `backend-mlir-compiler/` directory contains the MLIR-based compilation backend:
- `proto/` - Protobuf metadata definitions
- `level-1-pass/` - Compile-time MLIR compilation pass (uses `alwayslink=True`)
- `custom-op-mlir/` - Runtime custom op execution (uses `alwayslink=True`)
- `common/` - Shared utilities

## Dependencies

### Production Setup (Long Term)
- **MorphiZen**: Git submodule at `3rd-party/morphizen`
- **Bazel Registry**: https://github.com/ROCm/bazel-registry
- **Bazel C++ Toolchain**: From ROCm bazel-registry

### Local Development Setup (Current)
For local development, use `local_path_override` in MODULE.bazel:

```python
# Toolchain override for custom C++ toolchain with clang-cl support
local_path_override(
    module_name = "bazel_cpp_toolchain",
    path = "../../bazel-cpp-toolchain/bazel-cpp-toolchain-1",
)

# Local override for MorphiZen module
local_path_override(
    module_name = "morphizen",
    path = "../../MorphiZen-1",
)

# For ROCm bazel-registry modules, override with local clone
# registry(
#     name = "rocm_registry",
#     url = "file:///path/to/local/bazel-registry",
# )
```

### Required Dependencies
- **Protobuf**: v29.0
- **glog**: v0.7.1
- **rules_proto**: v7.0.2
- **LLVM/MLIR**: via llvm-project-overlay v22.1.3

## Known MorphiZen Issues (Fixed Locally)

### 1. Encryption Dependency
The MorphiZen `morphizen-core/BUILD.bazel` had an incorrect dependency:
- **Issue**: Line 386 referenced `"//encryption"` which doesn't exist
- **Fix**: Changed to `"//morphizen-foundation"` (encryption code is there)

### 2. Missing cmake/scripts BUILD
- **Issue**: `cmake/scripts/xxd.py` referenced but no BUILD.bazel
- **Fix**: Created `cmake/scripts/BUILD.bazel` with `exports_files(["xxd.py"])`

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
