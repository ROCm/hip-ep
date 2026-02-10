<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Implementation Plan: Issue #061 - Windows CI MSBuild Generator Compatibility

## Overview

Create a new Windows CI workflow that validates MorphiZen builds with auto-detected CMake generator (Visual Studio/MSBuild) while reusing cached dependencies from the Ninja-based workflow.

## Context

**Current Situation:**
- All Windows CI uses Ninja generator exclusively
- Users without Ninja see different behavior than CI
- MSBuild is default CMake generator on Windows
- No validation that MorphiZen builds with MSBuild

**Goal:**
Add parallel CI workflow that tests default Windows user experience without slowing down existing Ninja-based CI.

## Implementation Steps

### Step 1: Create New Workflow File

**File:** `.github/workflows/build_and_test_win_msbuild.yml`

**Base:** Copy from `build_and_test_win.yml` (lines 1-325)

**Changes to make:**

#### 1.1: Update Workflow Metadata (lines 5-18)

**Before:**
```yaml
name: build-and-test-win

on:
  workflow_dispatch:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  merge_group:
  push:
    branches:
      - main

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

**After:**
```yaml
name: build-and-test-win-msbuild

on:
  workflow_dispatch:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  merge_group:
  push:
    branches:
      - main

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

**Changes:** Only the workflow name (line 5)

#### 1.2: Keep Environment Variables Unchanged (lines 23-32)

**No changes needed:**
```yaml
env:
  MORPHIZEN_PREFIX: ${{ github.workspace }}/install
  MORPHIZEN_BUILD_DIR: ${{ github.workspace }}/cmake-build
  MORPHIZEN_WORKSPACE: ${{ github.workspace }}
  ONNXRUNTIME_VERSION: "1.23.2"
  LLVM_COMMIT: "f8cb7987c64dcffb72414a40560055cb717dbf74"
  PROTOBUF_VERSION: "v21.12"
  GTEST_VERSION: "v1.15.0"
  GLOG_VERSION: "v0.7.1"
  GSL_VERSION: "v4.0.0"
```

**Rationale:** Same dependency versions ensure cache compatibility

#### 1.3: Keep Dependency Steps Unchanged (lines 121-300)

**All these steps stay EXACTLY the same:**
- Cache Dependencies (lines 121-126)
- Download pre-built ONNXRuntime (lines 129-171)
- Checkout/Build LLVM/MLIR (lines 174-203)
- Checkout/Build protobuf (lines 206-226)
- Checkout/Build gtest (lines 229-248)
- Checkout/Build glog (lines 251-271)
- Checkout/Install GSL (lines 274-292)
- Save Dependencies Cache (lines 295-300)

**Rationale:**
- Dependencies are generator-agnostic (just libraries and headers)
- Reusing Ninja-built dependencies saves 45-55 minutes of CI time
- Cache key has no generator suffix - safe to share across workflows

**CRITICAL - Cache Compatibility:**
The cache key is:
```
deps-win-x64-release-mt-ort-$VERSION-llvm-$COMMIT-proto-$VERSION-gtest-$VERSION-glog-$VERSION-gsl-$VERSION
```

Notice: **No generator mentioned in key**

This means:
- ✅ Ninja workflow builds dependencies → saves to cache
- ✅ MSBuild workflow reads same cache → reuses dependencies
- ✅ Both workflows share same dependency cache
- ✅ No duplicate builds needed

#### 1.4: Update MorphiZen CMake Configure Step (lines 304-311)

**Before (Ninja workflow):**
```yaml
      - name: CMake Configure
        shell: cmd
        run: |
          cmake -G Ninja -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON ^
            -DCMAKE_PREFIX_PATH=${{ env.MORPHIZEN_PREFIX }}
```

**After (MSBuild workflow):**
```yaml
      - name: CMake Configure (MSBuild/Visual Studio auto-detected)
        shell: cmd
        run: |
          cmake -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON ^
            -DCMAKE_PREFIX_PATH=${{ env.MORPHIZEN_PREFIX }}
```

**Changes:**
1. **Removed `-G Ninja`** - Let CMake auto-detect (will use Visual Studio generator)
2. **Removed `-DCMAKE_BUILD_TYPE=Release`** - MSBuild is multi-config (Debug+Release in same tree)
3. **Updated step name** - Clarify this is testing MSBuild/Visual Studio

**Why remove CMAKE_BUILD_TYPE?**
- Visual Studio generator is **multi-config**: Both Debug and Release configs exist in same build tree
- CMAKE_BUILD_TYPE is for **single-config** generators only (Ninja, Makefile)
- With multi-config, you specify config at build time: `cmake --build ... --config Release`
- Setting CMAKE_BUILD_TYPE with multi-config generator causes a CMake warning

#### 1.5: Update MorphiZen Build Step (lines 313-316)

**Before (Ninja workflow):**
```yaml
      - name: CMake Build
        shell: cmd
        run: |
          ninja -C ${{ env.MORPHIZEN_BUILD_DIR }}
```

**After (MSBuild workflow):**
```yaml
      - name: CMake Build
        shell: cmd
        run: |
          cmake --build ${{ env.MORPHIZEN_BUILD_DIR }} --config Release --parallel
```

**Changes:**
1. **Changed from `ninja -C ...` to `cmake --build ...`** - Use portable CMake build command
2. **Added `--config Release`** - Required for multi-config generators (specifies which config to build)
3. **Added `--parallel`** - Enable parallel builds (equivalent to Ninja's default behavior)

**Why `cmake --build` instead of `msbuild`?**
- Portable across generators (works with Ninja, MSBuild, Make, etc.)
- CMake translates to appropriate native command (MSBuild.exe for Visual Studio generator)
- Recommended approach in CMake documentation

**MSBuild equivalent:**
```cmd
REM What CMake runs internally:
msbuild ${{ env.MORPHIZEN_BUILD_DIR }}\MorphiZen.sln /p:Configuration=Release /m
```

But `cmake --build` is cleaner.

#### 1.6: Keep Test Step Unchanged (lines 318-324)

**No changes needed:**
```yaml
      - name: Run Tests
        shell: cmd
        run: |
          cd ${{ env.MORPHIZEN_BUILD_DIR }}
          ctest -C Release --output-on-failure --timeout 120 -j4
```

**Rationale:**
- `ctest -C Release` works identically for both single-config and multi-config generators
- Test executables are in same location: `build/bin/Release/morphizen-unit-tests.exe`

### Step 2: Add Explanatory Comment

Add a comment at the top of the MorphiZen build section (before line 302) explaining the purpose:

```yaml
      # Build MorphiZen with auto-detected generator (Visual Studio/MSBuild)
      #
      # This workflow validates that MorphiZen builds successfully with the default
      # Windows CMake generator (Visual Studio/MSBuild), which many users have installed.
      # The Ninja-based workflow (build_and_test_win.yml) remains the primary CI for speed.
      #
      # Key differences from Ninja workflow:
      # - No -G flag: Auto-detect generator (defaults to Visual Studio on Windows)
      # - No CMAKE_BUILD_TYPE: Visual Studio is multi-config (Debug+Release in same tree)
      # - Build command: cmake --build with --config Release (instead of ninja)
      # - Dependencies: Reuse Ninja-built cache (generator-agnostic)
      #
      - name: CMake Configure (MSBuild/Visual Studio auto-detected)
        ...
```

### Step 3: Verify Complete Workflow File

**Final structure of `.github/workflows/build_and_test_win_msbuild.yml`:**

```yaml
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
name: build-and-test-win-msbuild

on:
  workflow_dispatch:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  merge_group:
  push:
    branches:
      - main

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

permissions:
  contents: read

env:
  MORPHIZEN_PREFIX: ${{ github.workspace }}/install
  MORPHIZEN_BUILD_DIR: ${{ github.workspace }}/cmake-build
  MORPHIZEN_WORKSPACE: ${{ github.workspace }}
  ONNXRUNTIME_VERSION: "1.23.2"
  LLVM_COMMIT: "f8cb7987c64dcffb72414a40560055cb717dbf74"
  PROTOBUF_VERSION: "v21.12"
  GTEST_VERSION: "v1.15.0"
  GLOG_VERSION: "v0.7.1"
  GSL_VERSION: "v4.0.0"

jobs:
  build-and-test:
    if: github.event_name == 'push' || github.event.pull_request.draft == false
    runs-on: windows-latest
    steps:
      - name: Checkout MorphiZen
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          submodules: true
          fetch-depth: 0
          lfs: true
          show-progress: false

      - name: Setup Python
        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6.0.0
        with:
          python-version: '3.12'

      - name: Setup MSVC
        uses: ilammy/msvc-dev-cmd@v1

      # Cache dependencies (identical to Ninja workflow - cache is generator-agnostic)
      - name: Cache Dependencies
        id: cache-deps
        uses: actions/cache@v4
        with:
          path: ${{ env.MORPHIZEN_PREFIX }}
          key: deps-win-x64-release-mt-ort-${{ env.ONNXRUNTIME_VERSION }}-llvm-${{ env.LLVM_COMMIT }}-proto-${{ env.PROTOBUF_VERSION }}-gtest-${{ env.GTEST_VERSION }}-glog-${{ env.GLOG_VERSION }}-gsl-${{ env.GSL_VERSION }}

      # Download pre-built ONNXRuntime if cache miss (identical to Ninja workflow)
      - name: Download pre-built ONNXRuntime
        if: steps.cache-deps.outputs.cache-hit != 'true'
        shell: powershell
        run: |
          # ... (copy entire step from build_and_test_win.yml lines 130-171)

      # Build LLVM/MLIR (identical to Ninja workflow)
      - name: Checkout LLVM
        if: steps.cache-deps.outputs.cache-hit != 'true'
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          repository: llvm/llvm-project
          ref: ${{ env.LLVM_COMMIT }}
          path: llvm-project
          show-progress: false

      - name: Build LLVM/MLIR
        if: steps.cache-deps.outputs.cache-hit != 'true'
        shell: cmd
        run: |
          # ... (copy entire step from build_and_test_win.yml lines 186-203)

      # Build protobuf, gtest, glog, GSL (all identical to Ninja workflow)
      # ... (copy steps from build_and_test_win.yml lines 206-292)

      # Save cache (identical to Ninja workflow)
      - name: Save Dependencies Cache
        if: steps.cache-deps.outputs.cache-hit != 'true'
        uses: actions/cache/save@v4
        with:
          path: ${{ env.MORPHIZEN_PREFIX }}
          key: deps-win-x64-release-mt-ort-${{ env.ONNXRUNTIME_VERSION }}-llvm-${{ env.LLVM_COMMIT }}-proto-${{ env.PROTOBUF_VERSION }}-gtest-${{ env.GTEST_VERSION }}-glog-${{ env.GLOG_VERSION }}-gsl-${{ env.GSL_VERSION }}

      # Build MorphiZen with auto-detected generator (MSBuild/Visual Studio)
      #
      # This workflow validates that MorphiZen builds successfully with the default
      # Windows CMake generator (Visual Studio/MSBuild), which many users have installed.
      # The Ninja-based workflow (build_and_test_win.yml) remains the primary CI for speed.
      #
      # Key differences from Ninja workflow:
      # - No -G flag: Auto-detect generator (defaults to Visual Studio on Windows)
      # - No CMAKE_BUILD_TYPE: Visual Studio is multi-config (Debug+Release in same tree)
      # - Build command: cmake --build with --config Release (instead of ninja)
      # - Dependencies: Reuse Ninja-built cache (generator-agnostic)
      #
      - name: CMake Configure (MSBuild/Visual Studio auto-detected)
        shell: cmd
        run: |
          cmake -B ${{ env.MORPHIZEN_BUILD_DIR }} -S . ^
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
            -Dmorphizen_ENABLE_UNIT_TEST=ON ^
            -DCMAKE_PREFIX_PATH=${{ env.MORPHIZEN_PREFIX }}

      - name: CMake Build
        shell: cmd
        run: |
          cmake --build ${{ env.MORPHIZEN_BUILD_DIR }} --config Release --parallel

      # Run tests (identical to Ninja workflow)
      - name: Run Tests
        shell: cmd
        run: |
          cd ${{ env.MORPHIZEN_BUILD_DIR }}
          ctest -C Release --output-on-failure --timeout 120 -j4
```

**Total lines:** ~340 (similar to original workflow)

## Testing Plan

### Local Testing (Before Committing)

**Not applicable** - Cannot test GitHub Actions workflows locally without third-party tools (act, nektos/act). Will verify after push.

### CI Testing (After Creating Workflow)

1. **Create test PR:**
   ```bash
   git checkout -b test/issue-061-msbuild-ci
   # Add workflow file
   git add .github/workflows/build_and_test_win_msbuild.yml
   git commit -m "ci: add MSBuild generator compatibility workflow"
   git push fork test/issue-061-msbuild-ci
   gh pr create --draft --title "Test: Issue #061 - MSBuild CI workflow"
   ```

2. **Verify workflow appears:**
   - Go to PR → "Checks" tab
   - Should see both "build-and-test-win" and "build-and-test-win-msbuild"

3. **Monitor workflow execution:**
   - Click on "build-and-test-win-msbuild" check
   - Verify "Cache Dependencies" step shows cache hit (reusing Ninja-built deps)
   - Verify "CMake Configure" step shows Visual Studio generator detected:
     ```
     -- Selecting Windows SDK version to target Windows 10.0.XXXXX...
     -- The CXX compiler identification is MSVC 19.XX.XXXXX
     -- Detecting CXX compiler ABI info - done
     -- Check for working CXX compiler: Visual Studio 17 2022 - skipped
     ```
   - Verify "CMake Build" step runs MSBuild successfully
   - Verify "Run Tests" step passes all tests

4. **Check build artifacts:**
   - Build outputs should be in: `build/bin/Release/morphizen-unit-tests.exe`
   - Not in: `build/bin/morphizen-unit-tests.exe` (Ninja location)

5. **Compare with Ninja workflow:**
   - Both should take similar time (since dependencies are cached)
   - Both should pass same tests
   - Build step might be slightly slower with MSBuild (expected)

### Manual Trigger Test

After merging, test manual workflow trigger:
```bash
# Via GitHub UI:
# 1. Go to Actions tab
# 2. Select "build-and-test-win-msbuild" workflow
# 3. Click "Run workflow" button
# 4. Select branch (main)
# 5. Click green "Run workflow" button
# 6. Monitor execution
```

## Potential Issues and Solutions

### Issue 1: Cache Miss on First Run

**Symptom:** MSBuild workflow runs before Ninja workflow, cache miss, builds all dependencies with Ninja

**Impact:** Low - dependencies still build correctly, just takes longer on first run

**Solution:** None needed - cache will be populated and reused on subsequent runs

### Issue 2: CMake Detects Ninja Instead of Visual Studio

**Symptom:** CMake configure output shows "Ninja" instead of "Visual Studio"

**Cause:** Ninja is in PATH on GitHub runners

**Solution:** Explicitly specify generator:
```yaml
cmake -G "Visual Studio 17 2022" -B ... -S .
```

**Decision:** Try auto-detect first. If Ninja is detected, switch to explicit generator in follow-up PR.

### Issue 3: MSBuild Fails with Runtime Library Mismatch

**Symptom:** Link error about mixing /MT and /MD

**Cause:** Dependencies built with different runtime than MorphiZen

**Solution:** Verify `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` is set (already in plan)

### Issue 4: Tests Fail Due to Missing DLLs

**Symptom:** `ctest` fails with "onnxruntime.dll not found"

**Cause:** DLLs not in PATH

**Solution:** DLLs are in `$MORPHIZEN_PREFIX/lib` - should be found via RPATH. If fails, add to PATH:
```yaml
      - name: Run Tests
        shell: cmd
        run: |
          set PATH=${{ env.MORPHIZEN_PREFIX }}\lib;%PATH%
          cd ${{ env.MORPHIZEN_BUILD_DIR }}
          ctest -C Release --output-on-failure --timeout 120 -j4
```

## Success Criteria Checklist

After implementation:

- [ ] Workflow file created: `.github/workflows/build_and_test_win_msbuild.yml`
- [ ] Workflow name is `build-and-test-win-msbuild`
- [ ] Workflow triggers on PR, push to main, workflow_dispatch
- [ ] Cache key matches Ninja workflow (generator-agnostic)
- [ ] All dependency steps identical to Ninja workflow
- [ ] MorphiZen configure step has NO `-G` flag
- [ ] MorphiZen configure step has NO `-DCMAKE_BUILD_TYPE`
- [ ] MorphiZen build step uses `cmake --build --config Release --parallel`
- [ ] Test step uses `ctest -C Release` (same as Ninja)
- [ ] Explanatory comment added before MorphiZen build
- [ ] Workflow appears in GitHub Actions UI
- [ ] Workflow reuses dependency cache from Ninja workflow
- [ ] CMake detects Visual Studio generator (verify in logs)
- [ ] Build completes successfully
- [ ] All tests pass

## File Summary

**Files to create:**
- `.github/workflows/build_and_test_win_msbuild.yml` (~340 lines)

**Files to modify:**
- None (purely additive change)

## Estimated Implementation Time

- **File creation:** 15 minutes (copy + modify)
- **Testing/verification:** 30 minutes (wait for CI, check logs)
- **Documentation:** 10 minutes (update issue when complete)

**Total:** ~1 hour

## Additional Notes

**Why this approach is safe:**
1. **No changes to existing workflows** - purely additive
2. **Reuses proven dependency builds** - only changes MorphiZen build
3. **Same test suite** - validates identical behavior
4. **Easy to revert** - just delete the file if issues arise

**Future improvements:**
1. Could add explicit generator flag if auto-detect chooses Ninja
2. Could add Debug build testing (multi-config supports both)
3. Could add ARM64 Windows testing (requires different runner)
