# FetchContent LLVM Integration

This document describes how morphizen-mlir integrates LLVM/MLIR using CMake's FetchContent mechanism, allowing builds without pre-installing LLVM.

## Overview

morphizen-mlir supports two ways to obtain LLVM/MLIR:

| Approach | Description | CMake Configure Time |
|----------|-------------|---------------------|
| **FetchContent** | Automatically downloads and builds LLVM inline | ~5-10 minutes (first time) |
| **Pre-installed** | Uses LLVM from `CMAKE_PREFIX_PATH` | ~30 seconds |

Both approaches work seamlessly with MorphiZen's `find_package(MLIR)` calls.

## How It Works

### 1. Detection Logic (cmake/llvm.cmake)

```cmake
# Try to find pre-installed LLVM/MLIR first
find_package(LLVM QUIET CONFIG)
find_package(MLIR QUIET CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
  message(STATUS "Found pre-installed LLVM and MLIR")
  set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
else()
  message(STATUS "LLVM/MLIR not found, will use FetchContent")
  set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)
  
  # Download and configure LLVM inline
  FetchContent_Declare(llvm-project
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG f8cb7987c64dcffb72414a40560055cb717dbf74
    ...)
  FetchContent_MakeAvailable(llvm-project)
  
  # Set include directories for downstream targets
  set(LLVM_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/llvm/include"
    "${llvm-project_BINARY_DIR}/include" ...)
  set(MLIR_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/mlir/include"
    "${llvm-project_BINARY_DIR}/tools/mlir/include" ...)
endif()
```

### 2. Using MLIR (level-1-pass-mlir/CMakeLists.txt)

```cmake
# find_package works for both pre-installed and FetchContent LLVM:
# - Pre-installed: imports targets via MLIRTargets.cmake
# - FetchContent: targets already exist, MLIRConfig.cmake skips import
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# MLIR targets work regardless of which approach is used
target_link_libraries(${LIB_NAME} PUBLIC MLIRIR MLIRFuncDialect ...)
```

**Note:** No conditional is needed! `find_package(MLIR)` works in both scenarios because LLVM's `MLIRConfig.cmake` is smart enough to skip importing targets when they already exist.

## Why find_package(MLIR) Works in MorphiZen

MorphiZen's `mlir-imp/CMakeLists.txt` contains:
```cmake
find_package(MLIR REQUIRED CONFIG)
```

This works even when using FetchContent because:

### Order of Operations

1. **cmake/llvm.cmake** runs first (via `include()` in deps.cmake)
2. `FetchContent_MakeAvailable(llvm-project)` configures LLVM as a subdirectory
3. All MLIR targets (MLIRIR, MLIRFuncDialect, etc.) are now defined in CMake scope
4. **Later**, MorphiZen is fetched and configured
5. When `mlir-imp/CMakeLists.txt` calls `find_package(MLIR)`:
   - CMake finds `MLIRConfig.cmake` in the build directory
   - The config file checks if targets are already defined
   - Since targets exist, it skips importing `MLIRTargets.cmake`

### LLVM's Smart Config Files

LLVM's `MLIRConfig.cmake` has conditional logic:

```cmake
# Simplified MLIRConfig.cmake logic
if(NOT TARGET MLIRIR)
  # Only import targets if not already defined
  include("${MLIR_CMAKE_DIR}/MLIRTargets.cmake")
endif()

# Set up MLIR variables and functions regardless
set(MLIR_INCLUDE_DIRS ...)
```

When LLVM is built via `add_subdirectory()` (which FetchContent uses internally):
- Targets are created directly in CMake scope
- `find_package(MLIR)` finds the config file
- Config file sees targets exist, skips the `MLIRTargets.cmake` include
- No error occurs

### Key Insight

| Scenario | find_package Behavior |
|----------|----------------------|
| **Pre-installed LLVM** | Imports targets via MLIRTargets.cmake |
| **FetchContent LLVM** | Targets already exist, import skipped |

Both scenarios result in working MLIR targets - they're just created differently.

## Pre-installing LLVM (Optional)

For faster cmake configure times, you can pre-build LLVM:

```batch
# Clone and checkout
git clone https://github.com/llvm/llvm-project.git ../llvm
cd ../llvm && git checkout f8cb7987c64dcffb72414a40560055cb717dbf74

# Configure
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release \
  -B ../build/llvm -S ../llvm/llvm \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_RTTI=ON

# Build and install
cmake --build ../build/llvm
cmake --install ../build/llvm
```

Then configure morphizen-mlir with:
```batch
cmake -S . -B build -DCMAKE_PREFIX_PATH=../local
```

## Historical Note: The Original Problem

Initially, FetchContent LLVM didn't work because:

1. **Error 1**: `find_package(MLIR)` couldn't find `MLIRTargets.cmake`
   - This file is only generated during `cmake --install`
   - FetchContent doesn't execute the install step

2. **Error 2**: MLIR headers not found
   - Missing MLIR source directory in include paths

The solution was to:
1. Set `LLVM_INCLUDE_DIRS` and `MLIR_INCLUDE_DIRS` explicitly after FetchContent
2. Rely on LLVM's smart config files that skip imports when targets already exist
3. Call `find_package(LLVM/MLIR)` unconditionally - it works in both scenarios!

## Related Files

| File | Purpose |
|------|---------|
| `cmake/llvm.cmake` | LLVM detection and FetchContent logic |
| `cmake/deps.cmake` | Includes llvm.cmake, enables MLIR backend |
| `level-1-pass-mlir/CMakeLists.txt` | Example of using find_package with FetchContent |
| `build_llvm.bat` | Script to pre-build LLVM (optional) |

## References

- [CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- [CMake find_package](https://cmake.org/cmake/help/latest/command/find_package.html)
- [LLVM CMake Documentation](https://llvm.org/docs/CMake.html)
