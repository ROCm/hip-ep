<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #061: Windows CI MSBuild Generator Compatibility

## Metadata
- **Type:** CI/Testing Infrastructure
- **Priority:** MEDIUM
- **Created:** 2026-02-10
- **Dependencies:** None (standalone CI improvement)

## Description

Create a separate Windows CI workflow that validates MorphiZen builds successfully with auto-detected CMake generator (Visual Studio/MSBuild) instead of Ninja.

## Problem

**Current State:**
- All Windows CI workflows use Ninja generator (`-G Ninja`)
- Many Windows developers don't have Ninja installed by default
- Default CMake behavior on Windows uses Visual Studio generator
- No CI validation that MorphiZen builds with MSBuild

**Why This Matters:**
- Users following documentation may not have Ninja installed
- Visual Studio generator is the default Windows CMake experience
- MSBuild has different behavior than Ninja (multi-config vs single-config)
- Compatibility issues with MSBuild won't be caught until user reports them

**Evidence:**
- `build_and_test_win.yml:187` - LLVM uses `-G Ninja`
- `build_and_test_win.yml:219` - protobuf uses `-G Ninja`
- `build_and_test_win.yml:242` - gtest uses `-G Ninja`
- `build_and_test_win.yml:264` - glog uses `-G Ninja`
- `build_and_test_win.yml:287` - GSL uses `-G Ninja`
- `build_and_test_win.yml:307` - MorphiZen uses `-G Ninja`
- `test-llvm-fetchcontent.yml:98,186` - Both test jobs use `-G Ninja`

## Solution

Create new CI workflow file `.github/workflows/build_and_test_win_msbuild.yml` that:

1. **Reuses cached dependencies** - Use same cache as Ninja workflow (all deps built with Ninja for speed)
2. **Auto-detect generator for MorphiZen** - Remove `-G Ninja` flag, let CMake choose Visual Studio/MSBuild
3. **Same triggers** - Run on PR, push to main, workflow_dispatch (same as existing workflow)
4. **Same configuration** - Release build, MultiThreaded runtime, unit tests enabled
5. **Multi-config aware** - MSBuild generator creates Debug/Release configs in same build tree

**Key Implementation Details:**

**Cache Compatibility:**
- Dependencies cache is generator-agnostic (just headers/libs)
- Cache key: `deps-win-x64-release-mt-ort-$VERSION-llvm-$COMMIT-...` (no generator in key)
- Safe to reuse Ninja-built dependencies with MSBuild-built MorphiZen

**Generator Differences:**

| Aspect | Ninja (existing) | MSBuild (new) |
|--------|------------------|---------------|
| Config Type | Single-config | Multi-config |
| Build Dir | Contains Release OR Debug | Contains both Release AND Debug |
| CMake Flag | `-DCMAKE_BUILD_TYPE=Release` | No CMAKE_BUILD_TYPE (uses `--config` at build time) |
| Build Command | `ninja -C build` | `cmake --build build --config Release` |
| Test Command | `ctest -C Release` | `ctest -C Release` (same) |

**Changes Required:**
1. Copy `build_and_test_win.yml` → `build_and_test_win_msbuild.yml`
2. Update workflow name: `build-and-test-win-msbuild`
3. Remove `-G Ninja` from MorphiZen CMake configure step
4. Remove `-DCMAKE_BUILD_TYPE=Release` from MorphiZen configure (multi-config generator)
5. Change build command: `ninja -C ...` → `cmake --build ... --config Release`
6. Keep ctest command: `ctest -C Release` (already correct)

## Files Affected

**New Files:**
- `.github/workflows/build_and_test_win_msbuild.yml` (new workflow, ~200 lines)

**No changes to existing files** - this is purely additive

## Verification

After creating the workflow:
1. Trigger workflow manually via GitHub Actions UI
2. Verify dependencies cache is reused (no rebuild of LLVM/protobuf/etc)
3. Verify MorphiZen configures with Visual Studio generator
4. Verify build succeeds with MSBuild
5. Verify all tests pass

## Success Criteria

- [ ] New workflow file created: `build_and_test_win_msbuild.yml`
- [ ] Workflow runs on PR, push to main, workflow_dispatch
- [ ] Workflow reuses dependency cache from Ninja workflow
- [ ] MorphiZen builds successfully with auto-detected generator (Visual Studio)
- [ ] All unit tests pass (same test coverage as Ninja workflow)
- [ ] Workflow appears in GitHub Actions tab
- [ ] CI passes on test PR

## Additional Context

**Why not replace Ninja workflow?**
- Ninja is faster for CI (single-config, parallel by default)
- Industry standard for CI (LLVM, Chromium use Ninja in CI)
- Keep both: Ninja for speed, MSBuild for compatibility

**Why only MorphiZen with MSBuild?**
- Dependencies are generator-agnostic (just headers/libs)
- Building all deps with MSBuild would 2-3x CI time
- Main goal is validating MorphiZen code builds with default Windows generator

**Alternative considered:**
- Add as job within existing workflow - Rejected because separate file is clearer and easier to maintain
