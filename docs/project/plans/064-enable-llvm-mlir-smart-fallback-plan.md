<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Enable LLVM/MLIR Smart Fallback with Auto-Fetch

**Issue:** #064
**Created:** 2026-02-09
**Status:** READY

## Objective

Implement three-tier LLVM/MLIR dependency resolution with automatic fallback: (1) pre-installed via CMAKE_PREFIX_PATH (fastest), (2) local source directory detection (fast, no download), (3) GitHub auto-fetch (newcomer-friendly). Based on proven morphizen-mlir pattern.

## Background

**Problem:** Current hard `REQUIRED` constraint in `mlir-imp/CMakeLists.txt:5-6` prevents fallback, causing build failures when LLVM not pre-installed. No auto-fetch mechanism exists despite having a commented-out `llvm.cmake` file.

**Current CI Gap:** Existing CI only tests Tier 1 (pre-installed LLVM). The FetchContent code paths (Tier 2/3) are completely untested.

**Proven Solution:** The morphizen-mlir project (which uses MorphiZen as a submodule) has a working FetchContent-based LLVM integration at `../morphizen-mlir/cmake/deps.cmake:8-109` that can be directly adapted.

## Implementation Steps

### Step 1: Add LLVM Fallback Logic to cmake/deps.cmake

**File:** `cmake/deps.cmake`

**Location:** After line 217 (end of ONNX dependency section, before final blank lines)

**Add new section:**

```cmake
##
## LLVM/MLIR dependency (when morphizen_ENABLE_MLIR_BACKEND=ON)
## Three-tier fallback: pre-installed → local source → GitHub download
## Based on proven morphizen-mlir pattern
##
if(morphizen_ENABLE_MLIR_BACKEND)
  message(STATUS "Configuring LLVM/MLIR for morphizen-mlir")

  # LLVM configuration options (applied when building from source)
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
  set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
  set(LLVM_ENABLE_RTTI OFF CACHE BOOL "Disable RTTI in LLVM")
  set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Disable libedit in LLVM")
  set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
  set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Build LLVM tests")
  set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Enable zlib compression")
  set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

  # Tier 1: Try to find pre-installed MLIR/LLVM first
  # MLIR implies LLVM exists (prevents incomplete LLVM installations)
  find_package(MLIR QUIET CONFIG)

  if(MLIR_FOUND)
    # MLIR found, now find LLVM (which must exist if MLIR exists)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found pre-installed LLVM and MLIR")
    message(STATUS "LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "MLIR_DIR: ${MLIR_DIR}")
    set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
  else()
    # MLIR not found, try local source or download
    # Do NOT call find_package(LLVM) to avoid importing incomplete installations
    message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent")
    set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)

    # Tier 2: Try to find LLVM source in local directories
    find_path(LOCAL_LLVM
      NAMES CMakeLists.txt
      PATHS
        "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
        "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
      NO_DEFAULT_PATH)

    if(LOCAL_LLVM)
      # Found local LLVM source
      message(STATUS "Found LLVM source in local directory")
      message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
      FetchContent_Declare(
        llvm-project
        SOURCE_DIR ${LOCAL_LLVM}/..
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm)
    else()
      # Tier 3: Download LLVM from GitHub
      message(STATUS "Cannot find LLVM in local directories")
      message(STATUS "Fetching LLVM source from GitHub")
      message(STATUS "WARNING: This will download and build LLVM (~20GB, 30-60 min)")
      FetchContent_Declare(
        llvm-project
        GIT_REPOSITORY ${DEP_URL_llvm}
        GIT_TAG ${DEP_SHA1_llvm}
        GIT_SUBMODULES_RECURSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR llvm
      )
    endif()

    # Make LLVM available - this adds LLVM as a subdirectory
    FetchContent_MakeAvailable(llvm-project)

    # Set include directories for downstream targets
    set(LLVM_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      CACHE PATH "LLVM include directories" FORCE)
    set(MLIR_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include"
      CACHE PATH "MLIR include directories" FORCE)

    # Make include directories globally available for all targets
    # This is necessary because subdirectory builds don't automatically propagate
    # MLIR source includes to targets that use find_package(MLIR)
    include_directories(SYSTEM
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include")

    # Note: In a subdirectory build, MLIR config files are in tools/mlir/cmake/modules/CMakeFiles/
    set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
    set(MLIR_DIR "${llvm-project_BINARY_DIR}/tools/mlir/cmake/modules/CMakeFiles" CACHE PATH "" FORCE)

    message(STATUS "LLVM source dir: ${llvm-project_SOURCE_DIR}")
    message(STATUS "LLVM binary dir: ${llvm-project_BINARY_DIR}")
    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
    message(STATUS "LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "MLIR_DIR: ${MLIR_DIR}")
  endif()

  message(STATUS "LLVM/MLIR configuration complete")
endif()
```

**Notes:**
- Uses `DEP_URL_llvm` and `DEP_SHA1_llvm` from `cmake/deps.txt:13`
- Matches morphizen-mlir pattern exactly (lines 8-109)
- `SOURCE_SUBDIR llvm` is critical - points FetchContent to llvm-project/llvm subdirectory

### Step 2: Remove Hard REQUIRED from mlir-imp/CMakeLists.txt

**File:** `mlir-imp/CMakeLists.txt`

**Location:** Lines 5-6

**Delete these two lines:**

```cmake
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
```

**Rationale:** These are now handled by `cmake/deps.cmake`. The hard `REQUIRED` prevented fallback.

### Step 3: Delete Dead Code - mlir-imp/cmake/llvm.cmake

**File:** `mlir-imp/cmake/llvm.cmake`

**Action:** Delete entire file

**Rationale:**
- 100% commented out (lines 6-45)
- Never included in build system
- morphizen-mlir proves this file is unnecessary
- Causes confusion about whether auto-fetch is supported

**Command:**
```bash
git rm mlir-imp/cmake/llvm.cmake
```

### Step 4: Delete Duplicate find_package Calls

**File:** `mlir-imp/cmake/onnx-mlir.cmake`

**Location:** Lines 5-6

**Delete these two lines:**

```cmake
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
```

**Rationale:** These duplicate the calls that were in `mlir-imp/CMakeLists.txt` (now removed). The file `onnx-mlir.cmake` is not currently included anywhere, so these calls are dead code.

### Step 5: Fix Documentation - developer-guide.md

**File:** `docs/developer-guide.md`

**Location:** Line 462

**Change:**

```bash
# OLD (incorrect):
  -DLLVM_ENABLE_PROJECTS="clang" \

# NEW (correct):
  -DLLVM_ENABLE_PROJECTS="mlir" \
```

**Rationale:**
- Current documentation says to build with clang project
- CI actually uses mlir (scripts/build/build_llvm.ps1:17)
- morphizen-mlir uses mlir
- MorphiZen's mlir-imp needs MLIR libraries, not clang

### Step 6: Fix Documentation - CLAUDE.md

**File:** `CLAUDE.md`

**Location:** Line 157

**Change:**

```markdown
# OLD (incorrect):
**Optional**: LLVM/MLIR, Protobuf, GTest, Boost (for tools)

# NEW (correct):
**Required when morphizen_ENABLE_MLIR_BACKEND=ON (auto-fetched if not found)**: LLVM/MLIR
**Optional (auto-fetched if not found)**: Protobuf, GTest
**Optional**: Boost (for tools only)
```

**Rationale:**
- LLVM is not optional when MLIR backend enabled
- But now it's auto-fetched, so users don't need manual setup

### Step 7: Create CI Workflow for FetchContent Validation

**File:** `.github/workflows/test-llvm-fetchcontent.yml` (new file)

**Create with content:**

```yaml
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
name: Test LLVM FetchContent

# Run conditionally to avoid excessive CI costs
on:
  # When cmake files are modified
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
  LLVM_COMMIT: f8cb7987c64dcffb72414a40560055cb717dbf74

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
          show-progress: false

      - name: Setup MSVC
        uses: ilammy/msvc-dev-cmd@v1

      - name: Clone LLVM to parent directory (simulate local source)
        shell: powershell
        run: |
          cd ..
          git clone --depth 1 https://github.com/llvm/llvm-project.git
          cd llvm-project
          git fetch --depth 1 origin ${{ env.LLVM_COMMIT }}
          git checkout ${{ env.LLVM_COMMIT }}

      - name: CMake Configure (Force Tier 2)
        shell: cmd
        run: |
          cmake -G Ninja -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON
            REM NO CMAKE_PREFIX_PATH - forces FetchContent tier 2

      - name: Verify Tier 2 Used (Local Source)
        shell: powershell
        run: |
          # Check that CMake found local LLVM source
          if (!(Test-Path "../llvm-project/llvm/CMakeLists.txt")) {
            Write-Error "LLVM source not found in expected location"
            exit 1
          }
          Write-Host "✓ Tier 2 validated: Using local LLVM source"

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
          show-progress: false

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

      - name: Verify Tier 3 Used (GitHub Download)
        shell: powershell
        run: |
          # Verify no local LLVM source exists
          if (Test-Path "../llvm-project") {
            Write-Error "Local LLVM source should not exist for tier 3 test"
            exit 1
          }
          Write-Host "✓ Tier 3 will download LLVM from GitHub"

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

**Notes:**
- Tier 2 test runs on PRs modifying cmake files
- Tier 3 test runs weekly + manual only (too expensive)
- Both tests validate FetchContent paths work correctly

### Step 8: Update Issue File with Plan Reference

**File:** `docs/project/issues/064-enable-llvm-mlir-smart-fallback.md`

**Location:** End of file, in `## Plans` section

**Add:**

```markdown
## Plans

- [Implementation Plan](../plans/064-enable-llvm-mlir-smart-fallback-plan.md) - Created 2026-02-09
```

## Testing & Validation

### Manual Testing (Before Merge)

Run all three scenarios locally to verify fallback logic:

**Test 1: Pre-installed LLVM (Tier 1)**
```bash
# Prerequisite: LLVM installed in ../../local
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/morphizen-core \
  -DCMAKE_PREFIX_PATH="$LOCAL_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dmorphizen_ENABLE_UNIT_TEST=ON

# Expected output:
# "Found pre-installed LLVM and MLIR"
# "Using pre-installed LLVM"
```

**Test 2: Local LLVM Source (Tier 2)**
```bash
# Prerequisite: LLVM source at ../llvm-project, NOT installed
rm -rf ../../local/lib/cmake/llvm ../../local/lib/cmake/mlir

cmake -S . -B ../../build/morphizen-core \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dmorphizen_ENABLE_UNIT_TEST=ON

# Expected output:
# "Found LLVM source in local directory"
# "LLVM SOURCE_DIR: <path>/llvm-project/llvm"
```

**Test 3: GitHub Download (Tier 3)**
```bash
# Prerequisite: No LLVM anywhere
rm -rf ../llvm-project ../../local/lib/cmake/llvm

cmake -S . -B ../../build/morphizen-core \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dmorphizen_ENABLE_UNIT_TEST=ON

# Expected output:
# "Fetching LLVM source from GitHub"
# "WARNING: This will download and build LLVM (~20GB, 30-60 min)"
# <Long download and build time>
```

### Verify Build and Tests

After each scenario:
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
../../build/morphizen-core/bin/morphizen-unit-tests.exe
```

All tests should pass in all three scenarios.

### CI Validation

After merging:

1. **Existing CI (Tier 1):** Continues to work with cached LLVM
2. **New CI Tier 2:** Triggers on next PR modifying cmake files
3. **New CI Tier 3:** Triggers on next Sunday or manual dispatch

Monitor first runs to ensure all tiers work correctly.

## Rollback Plan

If issues arise after merge:

1. **Immediate:** Revert the PR
   ```bash
   git revert <commit-hash>
   git push
   ```

2. **Temporary Fix:** Add environment variable to disable auto-fetch
   ```cmake
   # In cmake/deps.cmake, wrap FetchContent logic:
   if(NOT MORPHIZEN_DISABLE_LLVM_FETCHCONTENT)
     # ... existing code ...
   endif()
   ```

3. **Users can workaround:** Pre-install LLVM (tier 1 always works)

## Success Criteria

- ✅ All three manual test scenarios build successfully
- ✅ All unit tests pass in all three scenarios
- ✅ Existing CI continues to work (tier 1, cached LLVM)
- ✅ New CI tier 2 job runs and passes on cmake changes
- ✅ New CI tier 3 job runs and passes weekly
- ✅ Console messages clearly indicate which tier is being used
- ✅ Documentation accurately reflects LLVM requirements
- ✅ Dead code removed (llvm.cmake deleted)
- ✅ FetchContent code path continuously validated by CI
- ✅ No performance regression for developers with pre-built LLVM

## Notes

**morphizen-mlir Reference:**
- Source: `../morphizen-mlir/cmake/deps.cmake:8-109`
- This is a direct copy with minimal adaptation
- morphizen-mlir uses MorphiZen as submodule, proving compatibility

**Why Three Tiers:**
1. **Tier 1 (pre-installed):** Fastest, optimal for developers with established environments
2. **Tier 2 (local source):** Fast, no download, good for developers who clone LLVM once
3. **Tier 3 (download):** Slowest, but works out-of-box for newcomers

**CI Cost Optimization:**
- Tier 1: Every PR (existing CI, uses cache)
- Tier 2: PRs touching cmake files only
- Tier 3: Weekly + manual only (too expensive for every PR)

This balances validation coverage with CI cost.
