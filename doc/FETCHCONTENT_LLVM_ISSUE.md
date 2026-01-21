# FetchContent LLVM and find_package(MLIR) Compatibility Issue

## Problem Overview

In the morphizen-mlir project, we attempted to use CMake's `FetchContent` to automatically fetch and build LLVM/MLIR to simplify the user build process. However, this approach is incompatible with MorphiZen's mlir-imp component, causing build failures.

## Error Messages

### Error 1: MLIRConfig.cmake Cannot Find MLIRTargets.cmake

```
CMake Error at D:/Develop/m/build/morphizen-mlir/_deps/llvm-build/tools/mlir/cmake/modules/CMakeFiles/MLIRConfig.cmake:37 (include):
  include could not find requested file:

    D:/Develop/m/build/morphizen-mlir/_deps/llvm-build/tools/mlir/lib/cmake/mlir/MLIRTargets.cmake
Call Stack (most recent call first):
  D:/Develop/m/MorphiZen/mlir-imp/CMakeLists.txt:6 (find_package)
```

### Error 2: MLIR Header Files Not Found

```
fatal error C1083: Cannot open include file: 'mlir/IR/BuiltinDialect.h': No such file or directory
```

Compiler include paths:
```
-ID:\Develop\m\llvm\llvm\include 
-ID:\Develop\m\build\morphizen-mlir\_deps\llvm-build\include 
-ID:\Develop\m\build\morphizen-mlir\_deps\llvm-build\tools\mlir\include
```

Missing: `-ID:\Develop\m\llvm\mlir\include` (MLIR source headers)

## Root Cause Analysis

### 1. How FetchContent Works

```cmake
FetchContent_Declare(llvm SOURCE_DIR ${LOCAL_LLVM})
FetchContent_MakeAvailable(llvm)
```

What `FetchContent_MakeAvailable` does:
- Adds LLVM as a subdirectory to the build tree
- Configures LLVM's CMake (generates build.ninja, etc.)
- **Does NOT** execute `cmake --install`
- **Does NOT** generate CMake config files in install directory

### 2. What find_package(MLIR) Requires

MorphiZen's mlir-imp/CMakeLists.txt:
```cmake
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
```

`find_package(... CONFIG)` requires:
- `LLVMConfig.cmake` and `MLIRConfig.cmake` files
- `LLVMTargets.cmake` and `MLIRTargets.cmake` files
- These files are only generated during `cmake --install`

### 3. Why FetchContent Is Not Sufficient

| Phase | FetchContent | Pre-installed LLVM |
|-------|-------------|-------------------|
| Configure | ✅ Generates build.ninja | ✅ Generates build.ninja |
| Build | ⚠️ Deferred until needed | ✅ Full build |
| Install | ❌ Not executed | ✅ Generates *Config.cmake |
| find_package | ❌ Missing *Targets.cmake | ✅ Can find |

## Attempted Solutions

### Solution 1: Set LLVM_DIR and MLIR_DIR

```cmake
set(LLVM_DIR "${llvm_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "Path to LLVM CMake files" FORCE)
set(MLIR_DIR "${llvm_BINARY_DIR}/tools/mlir/cmake/modules/CMakeFiles" CACHE PATH "Path to MLIR CMake files" FORCE)
```

**Result:** ❌ Failed
- `MLIRConfig.cmake` exists in `tools/mlir/cmake/modules/CMakeFiles/`
- But it tries to `include(MLIRTargets.cmake)`, which doesn't exist

### Solution 2: Manually Set MLIR_INCLUDE_DIRS

```cmake
set(MLIR_INCLUDE_DIRS 
  "${llvm_SOURCE_DIR}/../mlir/include"      # MLIR source headers
  "${llvm_BINARY_DIR}/tools/mlir/include"   # MLIR generated headers
  CACHE PATH "MLIR include directories" FORCE)
```

**Result:** ❌ Partial success
- Solved header file path issue
- But `find_package(MLIR)` still fails (missing MLIRTargets.cmake)

### Solution 3: Force Build LLVM

```cmake
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${llvm_BINARY_DIR} --target install
  WORKING_DIRECTORY ${llvm_BINARY_DIR}
  RESULT_VARIABLE LLVM_BUILD_RESULT
)
```

**Result:** ❌ Failed
- `execute_process` runs during configuration phase
- LLVM's build system is not fully initialized yet
- Error: `could not load cache`

## Final Solution

### ✅ Pre-build LLVM Using build_llvm.bat

**Reason:**
- `MLIRTargets.cmake` is only generated after `cmake --install`
- FetchContent cannot trigger the install step
- LLVM must be manually built and installed

**Implementation:**

1. **build_llvm.bat Script**
```batch
# Clone LLVM
git clone https://github.com/llvm/llvm-project.git D:/Develop/m/llvm

# Checkout specific commit
cd D:/Develop/m/llvm
git checkout f8cb7987c64dcffb72414a40560055cb717dbf74

# Configure
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -B D:/Develop/m/build/llvm -S D:/Develop/m/llvm/llvm \
  -DCMAKE_INSTALL_PREFIX=D:/Develop/m/local \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON

# Build
cmake --build D:/Develop/m/build/llvm

# Install (Critical step!)
cmake --install D:/Develop/m/build/llvm
```

2. **cmake/llvm.cmake Detection Logic**
```cmake
# First try to find pre-installed LLVM/MLIR
find_package(LLVM QUIET CONFIG)
find_package(MLIR QUIET CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
  message(STATUS "Found pre-installed LLVM and MLIR")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
else()
  # If not found, provide clear error message
  message(FATAL_ERROR 
    "LLVM/MLIR not found. Please run build_llvm.bat first.\n"
    "Reason: MorphiZen's mlir-imp uses find_package(MLIR) which requires\n"
    "MLIRTargets.cmake file that is only generated during 'cmake --install'.")
endif()
```

## Debugging Steps

### Step 1: Verify FetchContent Configured LLVM

```bash
# Check if LLVM build directory exists
Test-Path "d:/Develop/m/build/morphizen-mlir/_deps/llvm-build"
# Output: True
```

### Step 2: Locate MLIRConfig.cmake

```bash
Get-ChildItem "d:/Develop/m/build/morphizen-mlir/_deps/llvm-build" -Recurse -Filter "MLIRConfig.cmake"
# Output: D:\Develop\m\build\morphizen-mlir\_deps\llvm-build\tools\mlir\cmake\modules\CMakeFiles\MLIRConfig.cmake
```

**Finding:** MLIRConfig.cmake is in `CMakeFiles` subdirectory, not in standard location.

### Step 3: Check if MLIRTargets.cmake Exists

```bash
Test-Path "d:/Develop/m/build/morphizen-mlir/_deps/llvm-build/lib/cmake/mlir/MLIRTargets.cmake"
# Output: False
```

**Conclusion:** `MLIRTargets.cmake` doesn't exist because install step was not executed.

### Step 4: Check MLIR Header File Location

```bash
Test-Path "d:/Develop/m/llvm/mlir/include/mlir/IR/BuiltinDialect.h"
# Output: True
```

**Finding:** MLIR source headers exist, but this path is not included in compile command.

### Step 5: Run build_llvm.bat

```bash
cd d:/Develop/m/morphizen-mlir
.\build_llvm.bat
```

**Result:** ✅ Success
- LLVM build completed
- Installed to `D:/Develop/m/local`
- Generated `LLVMConfig.cmake`, `MLIRConfig.cmake`, `MLIRTargets.cmake`

### Step 6: Verify Installed LLVM

```bash
Test-Path "d:/Develop/m/local/lib/cmake/llvm/LLVMConfig.cmake"
# Output: True

Test-Path "d:/Develop/m/local/lib/cmake/mlir/MLIRConfig.cmake"
# Output: True

Test-Path "d:/Develop/m/local/lib/cmake/mlir/MLIRTargets.cmake"
# Output: True
```

**Conclusion:** Pre-installed LLVM contains all required CMake files.

### Step 7: Clean Build Directory and Reconfigure

```bash
Remove-Item -Recurse -Force "d:/Develop/m/build/morphizen-mlir"
.\build.bat
```

**Result:** ✅ CMake configuration successful
```
-- Found pre-installed LLVM and MLIR
-- LLVM_DIR: D:/Develop/m/local/lib/cmake/llvm
-- MLIR_DIR: D:/Develop/m/local/lib/cmake/mlir
```

## Technical Details

### MLIRConfig.cmake Content

FetchContent-generated `MLIRConfig.cmake` (in CMakeFiles):
```cmake
# Line 37
include("${MLIR_CMAKE_DIR}/MLIRTargets.cmake")
```

This file expects `MLIRTargets.cmake` at relative path `lib/cmake/mlir/`, but FetchContent doesn't generate this file.

### Why FetchContent Doesn't Generate Targets Files

CMake's `*Targets.cmake` files are generated by `install(EXPORT ...)` command:

```cmake
# In LLVM's CMakeLists.txt
install(EXPORT LLVMExports
  FILE LLVMTargets.cmake
  NAMESPACE LLVM::
  DESTINATION lib/cmake/llvm)
```

This command only runs when executing `cmake --install`.

## Conclusion

**FetchContent cannot fully replace pre-installed LLVM/MLIR** because:

1. ✅ FetchContent can configure and build LLVM
2. ❌ FetchContent does not execute install step
3. ❌ No install means no `*Targets.cmake` files
4. ❌ `find_package(MLIR CONFIG)` requires these files

**Recommended Approach:**
- Use `build_llvm.bat` to pre-build and install LLVM/MLIR
- Detect installed LLVM in `cmake/llvm.cmake`
- Provide clear error message if not found

## Related Files

- `cmake/llvm.cmake` - LLVM detection and configuration logic
- `build_llvm.bat` - LLVM build and install script
- `D:/Develop/m/MorphiZen/mlir-imp/CMakeLists.txt` - Code using find_package(MLIR)

## References

- CMake FetchContent: https://cmake.org/cmake/help/latest/module/FetchContent.html
- CMake find_package: https://cmake.org/cmake/help/latest/command/find_package.html
- LLVM CMake: https://llvm.org/docs/CMake.html
