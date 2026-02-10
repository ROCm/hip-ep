<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #064: Enable LLVM/MLIR Smart Fallback with Auto-Fetch

## Metadata
- **Type:** Feature
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Implement smart LLVM/MLIR dependency resolution with three-tier fallback: (1) pre-installed via CMAKE_PREFIX_PATH, (2) local source directory detection, (3) automatic download from GitHub. This eliminates the hard REQUIRED constraint that currently prevents newcomers from building MorphiZen with MLIR backend enabled, while maintaining optimal performance for developers with pre-built LLVM.

## Problem

**Current design:**
```cmake
# mlir-imp/CMakeLists.txt:5-6
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
```

**Why this is problematic:**

1. **Hard failure when LLVM not found** - Build fails immediately if LLVM not pre-installed, with no fallback or auto-fetch mechanism
2. **Poor newcomer experience** - New contributors must manually build LLVM (30-60 minutes) before building MorphiZen
3. **Inconsistent with other dependencies** - glog, protobuf, GTest all have smart fallback with FetchContent, but LLVM does not
4. **Dead code confusion** - `mlir-imp/cmake/llvm.cmake` exists but is 100% commented out (lines 6-45), causing confusion about whether auto-fetch is supported
5. **Documentation inaccuracy** - CLAUDE.md:157 claims "Optional: LLVM/MLIR" but LLVM is actually REQUIRED when `morphizen_ENABLE_MLIR_BACKEND=ON` (default)
6. **Configuration mismatch** - developer-guide.md:462 documents `LLVM_ENABLE_PROJECTS="clang"` but CI uses `"mlir"` (scripts/build/build_llvm.ps1:17)

**Code locations:**
- `mlir-imp/CMakeLists.txt:5-6` - Hard REQUIRED constraint with no fallback
- `mlir-imp/cmake/llvm.cmake:6-45` - Commented-out FetchContent code (dead code)
- `cmake/deps.cmake:217` - Other deps have fallback, LLVM missing
- `cmake/deps.txt:13` - LLVM commit defined but never used for auto-fetch
- `docs/developer-guide.md:462` - Wrong LLVM_ENABLE_PROJECTS value
- `CLAUDE.md:157` - Incorrect "Optional" claim

**Evidence from morphizen-mlir:**
The sister project morphizen-mlir (which uses MorphiZen as a submodule) has a proven working FetchContent-based LLVM integration at `../morphizen-mlir/cmake/deps.cmake:8-109` that implements the exact pattern needed.

## Solution

**Proposed design:**

Adopt the proven morphizen-mlir pattern with three-tier fallback:

```cmake
# cmake/deps.cmake (add after line 217, ONNX dependency)

if(morphizen_ENABLE_MLIR_BACKEND)
  # LLVM configuration options
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
  set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
  set(LLVM_ENABLE_RTTI OFF CACHE BOOL "Disable RTTI in LLVM")
  set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Disable libedit in LLVM")
  set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
  set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Build LLVM tests")
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Enable zlib compression")
  set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

  # Tier 1: Try pre-installed MLIR/LLVM first
  find_package(MLIR QUIET CONFIG)

  if(MLIR_FOUND)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Using pre-installed LLVM: ${LLVM_DIR}")
    message(STATUS "Using pre-installed MLIR: ${MLIR_DIR}")
  else()
    # Tier 2: Try to find LLVM source in local directories
    find_path(LOCAL_LLVM
      NAMES CMakeLists.txt
      PATHS
        "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
      NO_DEFAULT_PATH)

    if(LOCAL_LLVM)
      message(STATUS "Found LLVM source in local directory: ${LOCAL_LLVM}")
      FetchContent_Declare(
        llvm-project
        SOURCE_DIR ${LOCAL_LLVM}/..
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm)
    else()
      # Tier 3: Download LLVM from GitHub
      message(STATUS "LLVM not found locally, downloading from GitHub")
      message(STATUS "WARNING: This will download and build LLVM (~20GB, 30-60 min)")
      FetchContent_Declare(
        llvm-project
        GIT_REPOSITORY ${DEP_URL_llvm}
        GIT_TAG ${DEP_SHA1_llvm}
        GIT_SUBMODULES_RECURSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm)
    endif()

    # Build LLVM inline
    FetchContent_MakeAvailable(llvm-project)

    # Set paths for downstream targets
    set(LLVM_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      CACHE PATH "LLVM include directories" FORCE)
    set(MLIR_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include"
      CACHE PATH "MLIR include directories" FORCE)

    include_directories(SYSTEM
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include")

    set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
    set(MLIR_DIR "${llvm-project_BINARY_DIR}/tools/mlir/cmake/modules/CMakeFiles" CACHE PATH "" FORCE)
  endif()
endif()
```

**Approach:**

1. **Add LLVM fallback logic to cmake/deps.cmake** (after line 217)
   - Copy pattern from morphizen-mlir/cmake/deps.cmake:8-109
   - Wrap in `if(morphizen_ENABLE_MLIR_BACKEND)` condition
   - Use existing `DEP_URL_llvm` and `DEP_SHA1_llvm` from cmake/deps.txt:13

2. **Remove hard REQUIRED from mlir-imp/CMakeLists.txt** (lines 5-6)
   - Delete `find_package(LLVM REQUIRED CONFIG)`
   - Delete `find_package(MLIR REQUIRED CONFIG)`
   - These are now handled by cmake/deps.cmake

3. **Delete dead code: mlir-imp/cmake/llvm.cmake**
   - Entire file is unused (100% commented out)
   - morphizen-mlir proves this file is unnecessary

4. **Delete duplicate find_package calls**
   - mlir-imp/cmake/onnx-mlir.cmake lines 5-6 duplicate the CMakeLists.txt calls

5. **Fix documentation errors**
   - docs/developer-guide.md:462 - Change `LLVM_ENABLE_PROJECTS="clang"` → `"mlir"`
   - CLAUDE.md:157 - Change "Optional: LLVM/MLIR" → "Required when MLIR enabled (auto-fetched)"

6. **Test three scenarios:**
   - Pre-installed LLVM (existing workflow, should still work)
   - Local LLVM source at ../llvm-project (should auto-detect)
   - No LLVM at all (should download and build)

**Benefits:**

- ✅ **Newcomer-friendly** - Auto-downloads LLVM if not found, no manual setup required
- ✅ **Developer-optimized** - Reuses pre-installed or local LLVM source (no download)
- ✅ **CI-compatible** - Works with cached LLVM (tier 1) or from-scratch builds (tier 3)
- ✅ **Proven working** - Exact pattern used successfully in morphizen-mlir project
- ✅ **Consistent** - Matches glog/protobuf/GTest fallback pattern
- ✅ **Low risk** - morphizen-mlir uses MorphiZen as submodule, proving compatibility
- ✅ **Removes dead code** - Eliminates confusing commented-out llvm.cmake
- ✅ **Fixes documentation** - Corrects inaccuracies about LLVM being optional

## Plans

- [Implementation Plan](../plans/064-enable-llvm-mlir-smart-fallback-plan.md) - Created 2026-02-09

## Evidence

**morphizen-mlir proof of concept:**
The sister project at `../morphizen-mlir` uses MorphiZen as a git submodule (3rd-party/morphizen) and successfully builds it using the FetchContent LLVM pattern. This proves:
- The approach works with MorphiZen's mlir-imp component
- No compatibility issues exist
- The exact code can be copied with minimal adaptation

**CI validation:**
Current CI builds LLVM using the correct options (scripts/build/build_llvm.ps1:17):
- `LLVM_ENABLE_PROJECTS=mlir` (NOT clang)
- `LLVM_TARGETS_TO_BUILD=host` (NOT X86)
- Matches morphizen-mlir configuration exactly

**CRITICAL: CI Testing Gap Identified**

Current CI (`.github/workflows/build_and_test_win.yml:173-311`) does NOT test the FetchContent fallback path:

1. **What CI does:** Builds LLVM from source → Installs to MORPHIZEN_PREFIX → Caches installation → MorphiZen uses via CMAKE_PREFIX_PATH (line 311)
2. **What CI tests:** Only Tier 1 (pre-installed LLVM via find_package)
3. **What CI does NOT test:** Tier 2 (local source) or Tier 3 (GitHub download) - the entire FetchContent code path

**Risk:** FetchContent implementation could be broken/untested until a user encounters it.

**Mitigation:** Add conditional CI job to validate FetchContent path (see Testing Plan section below).

## Files to Modify

### Add LLVM fallback logic:
- `cmake/deps.cmake` - Add ~100 lines after line 217 (ONNX dependency section)

### Remove hard REQUIRED constraint:
- `mlir-imp/CMakeLists.txt` - Delete lines 5-6 (find_package calls)

### Delete dead code:
- `mlir-imp/cmake/llvm.cmake` - Delete entire file (100% commented out)
- `mlir-imp/cmake/onnx-mlir.cmake` - Delete lines 5-6 (duplicate find_package)

### Fix documentation:
- `docs/developer-guide.md` - Fix line 462 (LLVM_ENABLE_PROJECTS value)
- `CLAUDE.md` - Fix line 157 (LLVM optional claim)

### Add CI validation:
- `.github/workflows/test-llvm-fetchcontent.yml` - New conditional CI job to validate FetchContent path

## Estimated Effort

**3 hours** - Direct copy from proven morphizen-mlir implementation + CI job

- 30 min: Copy LLVM fallback logic to cmake/deps.cmake
- 15 min: Remove REQUIRED from mlir-imp, delete dead files
- 30 min: Fix documentation
- 30 min: Create CI workflow for FetchContent testing
- 1h 15min: Test three scenarios (pre-installed, local source, download)

## Testing Plan

### Manual Testing (Before Merge)

**Scenario 1: Pre-installed LLVM (existing workflow)**
```bash
# LLVM already in ../../local
cmake -S . -B ../../build/morphizen-core -DCMAKE_PREFIX_PATH=$(cd ../../local && pwd)
# Should use tier 1: pre-installed LLVM
```

**Scenario 2: Local LLVM source**
```bash
# LLVM source exists at ../llvm-project but not installed
rm -rf ../../local/lib/cmake/llvm
cmake -S . -B ../../build/morphizen-core
# Should use tier 2: local source (no download)
```

**Scenario 3: No LLVM at all**
```bash
# No LLVM anywhere
rm -rf ../llvm-project ../../local/lib/cmake/llvm
cmake -S . -B ../../build/morphizen-core
# Should use tier 3: download from GitHub (slow first build)
```

### Automated CI Testing (Ongoing Validation)

Create `.github/workflows/test-llvm-fetchcontent.yml` to validate FetchContent path:

```yaml
name: Test LLVM FetchContent

# Run conditionally to avoid excessive CI costs
on:
  # When cmake/deps.cmake is modified
  pull_request:
    paths:
      - 'cmake/deps.cmake'
      - 'cmake/deps.txt'
      - 'mlir-imp/CMakeLists.txt'
  # Weekly scheduled run
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday at midnight UTC
  # Manual trigger
  workflow_dispatch:

permissions:
  contents: read

env:
  MORPHIZEN_BUILD_DIR: ${{ github.workspace }}/cmake-build
  # NO MORPHIZEN_PREFIX - force FetchContent fallback

jobs:
  test-fetchcontent-tier2:
    name: Test Tier 2 (Local Source)
    runs-on: windows-latest
    steps:
      - name: Checkout MorphiZen
        uses: actions/checkout@v4
        with:
          submodules: true
          lfs: true

      - name: Setup MSVC
        uses: ilammy/msvc-dev-cmd@v1

      - name: Clone LLVM to parent directory (simulate local source)
        run: |
          cd ..
          git clone --depth 1 https://github.com/llvm/llvm-project.git
          cd llvm-project
          git fetch --depth 1 origin f8cb7987c64dcffb72414a40560055cb717dbf74
          git checkout f8cb7987c64dcffb72414a40560055cb717dbf74

      - name: CMake Configure (Force Tier 2)
        shell: cmd
        run: |
          cmake -G Ninja -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON
            REM NO CMAKE_PREFIX_PATH - forces FetchContent tier 2

      - name: Verify Tier 2 Used
        shell: cmd
        run: |
          REM Check CMake output mentions local LLVM source
          cmake -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . -LAH | findstr "llvm-project_SOURCE_DIR"

      - name: CMake Build
        shell: cmd
        run: |
          ninja -C ${{ env.MORPHIZEN_BUILD_DIR }}

      - name: Run Tests
        shell: cmd
        run: |
          cd ${{ env.MORPHIZEN_BUILD_DIR }}
          ctest -C Release --output-on-failure --timeout 120 -j4

  test-fetchcontent-tier3:
    name: Test Tier 3 (GitHub Download)
    runs-on: windows-latest
    # Only run on manual trigger or schedule (too expensive for every PR)
    if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'
    steps:
      - name: Checkout MorphiZen
        uses: actions/checkout@v4
        with:
          submodules: true
          lfs: true

      - name: Setup MSVC
        uses: ilammy/msvc-dev-cmd@v1

      - name: CMake Configure (Force Tier 3)
        shell: cmd
        run: |
          cmake -G Ninja -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON
            REM NO CMAKE_PREFIX_PATH, NO local llvm-project - forces tier 3 download

      - name: Verify Tier 3 Used
        shell: cmd
        run: |
          REM Check CMake output mentions downloading LLVM
          cmake -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . -LAH | findstr "GIT_REPOSITORY"

      - name: CMake Build
        shell: cmd
        run: |
          ninja -C ${{ env.MORPHIZEN_BUILD_DIR }}

      - name: Run Tests
        shell: cmd
        run: |
          cd ${{ env.MORPHIZEN_BUILD_DIR }}
          ctest -C Release --output-on-failure --timeout 120 -j4
```

**CI Strategy:**
- **Tier 2 test:** Runs on PRs modifying cmake files (validates local source detection)
- **Tier 3 test:** Runs weekly + manual only (validates GitHub download, too expensive for every PR)
- **Tier 1 test:** Already covered by existing build_and_test_win.yml

## Success Criteria

- ✅ All three manual test scenarios build successfully
- ✅ Existing CI continues to work (uses cached LLVM, tier 1)
- ✅ New CI workflow validates tier 2 (local source) on cmake changes
- ✅ New CI workflow validates tier 3 (download) on weekly schedule
- ✅ No performance regression for developers with pre-built LLVM
- ✅ Clear console messages indicating which tier is being used
- ✅ Documentation accurately reflects LLVM requirements
- ✅ No dead code remains (llvm.cmake deleted)
- ✅ FetchContent code path continuously validated by CI
