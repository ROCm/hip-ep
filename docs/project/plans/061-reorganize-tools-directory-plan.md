<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Implementation Plan: Issue #061 - Reorganize tools/ Directory

## Overview

Reorganize tools/ directory to contain only executables, and organize scripts into logical categories (debug, build, bazel, cmake).

**Estimated time:** 2-3 hours

## Prerequisites

- On main branch with no uncommitted changes
- Pre-commit hooks installed

## Phase 1: Create New Directory Structure

### Step 1.1: Create new directories

```bash
mkdir -p scripts/debug
mkdir -p scripts/build
mkdir -p scripts/bazel
mkdir -p cmake/scripts
```

**Verify:**
```bash
ls -la scripts/
ls -la cmake/
```

Should see: debug/, build/, bazel/ under scripts/, and scripts/ under cmake/

---

## Phase 2: Move Files to New Locations

### Step 2.1: Move debugging utilities

```bash
git mv tools/morphizen_check_version.py scripts/debug/
git mv tools/convert_onnx_to_external_data_mode.py scripts/debug/
```

**Verify:**
```bash
ls -la scripts/debug/
```

Should see: morphizen_check_version.py, convert_onnx_to_external_data_mode.py

### Step 2.2: Move CMake utilities

```bash
git mv tools/xxd.py cmake/scripts/
```

**Verify:**
```bash
ls -la cmake/scripts/
```

Should see: xxd.py

### Step 2.3: Move Bazel utilities

```bash
git mv tools/patch_wrapper.py scripts/bazel/
git mv tools/collect_pb.bzl scripts/bazel/
```

**Verify:**
```bash
ls -la scripts/bazel/
```

Should see: patch_wrapper.py, collect_pb.bzl

### Step 2.4: Move CI build scripts

```bash
git mv tools/run-external-command.ps1 scripts/build/
git mv tools/setup_msvc_env.ps1 scripts/build/
git mv tools/build_and_test.ps1 scripts/build/
git mv tools/build_llvm.ps1 scripts/build/
git mv tools/build_ort_and_deps.ps1 scripts/build/
```

**Verify:**
```bash
ls -la scripts/build/
```

Should see: 5 PowerShell scripts

---

## Phase 3: Delete Obsolete Files

### Step 3.1: Delete parse_cl_link_error.py

```bash
git rm tools/parse_cl_link_error.py
```

**Rationale:** AI can help manually parse linker errors when needed (rare occurrence)

### Step 3.2: Delete tools/BUILD.bazel

```bash
git rm tools/BUILD.bazel
```

**Rationale:** Will be replaced by scripts/bazel/BUILD.bazel

---

## Phase 4: Move Tool Directories into tools/

### Step 4.1: Move tool component directories

```bash
git mv graph-opt tools/
git mv onnx-grep tools/
git mv pattern-gen tools/
git mv tar tools/
```

**Verify:**
```bash
ls -la tools/
```

Should see: graph-opt/, onnx-grep/, pattern-gen/, tar/ (and nothing else!)

---

## Phase 5: Create New Files

### Step 5.1: Create scripts/debug/README.md

Create file: `scripts/debug/README.md`

```markdown
<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Debugging Utilities

Troubleshooting and debugging scripts for MorphiZen development.

## morphizen_check_version.py

Extract build information from MorphiZen DLL.

**Purpose:** Verify DLL build info, check component versions

**Usage:**
```bash
python morphizen_check_version.py <path_to_onnxruntime_vitisai_ep.dll>
```

**Example:**
```bash
python scripts/debug/morphizen_check_version.py ../../build/morphizen/bin/onnxruntime_vitisai_ep.dll
```

**Output:** Build info including version, commit hash, build date, etc.

---

## convert_onnx_to_external_data_mode.py

Convert ONNX models to external data format (separates large tensors into .data file).

**Purpose:** Debug large models, reduce .onnx file size

**Usage:**
```bash
python convert_onnx_to_external_data_mode.py <input.onnx> <output.onnx>
```

**Example:**
```bash
python scripts/debug/convert_onnx_to_external_data_mode.py model.onnx model_external.onnx
```

**Output:**
- `output.onnx` - Model structure
- `output.data` - External tensor data (for tensors > 128 bytes)

**Note:** ONNX library also provides similar functionality:
```bash
python -m onnx.tools.convert_model_to_external_data input.onnx output.onnx
```
```

**Verify:** File created with proper documentation

### Step 5.2: Create scripts/bazel/BUILD.bazel

Create file: `scripts/bazel/BUILD.bazel`

```python
##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "patch_wrapper",
    srcs = ["patch_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@my_pip_deps//patch",
    ],
)
```

**Verify:** File created with correct Bazel syntax

---

## Phase 6: Update References

### Step 6.1: Update morphizen-core/BUILD.bazel

**File:** `morphizen-core/BUILD.bazel`

**Change 1 - Line 8 (load statement):**

Before:
```python
load("//tools:collect_pb.bzl", "collect_pb")
```

After:
```python
load("//scripts/bazel:collect_pb.bzl", "collect_pb")
```

**Change 2 - Line 204-205 (patch_wrapper reference):**

Before:
```python
cmd = "$(location //tools:patch_wrapper) $(location patches/tar.h.force_align_1.patch) $(location @freebsd_tar_h//file) $(@D) -- --debug -v -p1 -d $(@D)",
tools = ["//tools:patch_wrapper"],
```

After:
```python
cmd = "$(location //scripts/bazel:patch_wrapper) $(location patches/tar.h.force_align_1.patch) $(location @freebsd_tar_h//file) $(@D) -- --debug -v -p1 -d $(@D)",
tools = ["//scripts/bazel:patch_wrapper"],
```

**Verify:**
```bash
grep "scripts/bazel" morphizen-core/BUILD.bazel
```

Should show both updated references

### Step 6.2: Update morphizen-demo/level-1-pass-relu-dq/cmake/generate_pattern_inc.cmake

**File:** `morphizen-demo/level-1-pass-relu-dq/cmake/generate_pattern_inc.cmake`

**Change - Line 9:**

Before:
```cmake
$<TARGET_FILE:Python3::Interpreter> ${morphizen_SOURCE_DIR}/tools/xxd.py
```

After:
```cmake
$<TARGET_FILE:Python3::Interpreter> ${morphizen_SOURCE_DIR}/cmake/scripts/xxd.py
```

**Verify:**
```bash
grep "cmake/scripts/xxd.py" morphizen-demo/level-1-pass-relu-dq/cmake/generate_pattern_inc.cmake
```

### Step 6.3: Update .github/actions/build-and-test/action.yml

**File:** `.github/actions/build-and-test/action.yml`

**Change - Line 78:**

Before:
```yaml
. ${{ github.workspace }}/tools/build_and_test.ps1
```

After:
```yaml
. ${{ github.workspace }}/scripts/build/build_and_test.ps1
```

**Verify:**
```bash
grep "scripts/build/build_and_test.ps1" .github/actions/build-and-test/action.yml
```

### Step 6.4: Update .github/actions/build-deps/action.yml

**File:** `.github/actions/build-deps/action.yml`

**Change 1 - Line 75:**

Before:
```yaml
. ${{ github.workspace}}/tools/build_ort_and_deps.ps1
```

After:
```yaml
. ${{ github.workspace}}/scripts/build/build_ort_and_deps.ps1
```

**Change 2 - Line 108:**

Before:
```yaml
. ${{ github.workspace}}/tools/build_llvm.ps1
```

After:
```yaml
. ${{ github.workspace}}/scripts/build/build_llvm.ps1
```

**Verify:**
```bash
grep "scripts/build" .github/actions/build-deps/action.yml
```

Should show both updated references

### Step 6.5: Update scripts/build/*.ps1 internal references

**Files to update:**
- `scripts/build/build_and_test.ps1`
- `scripts/build/build_llvm.ps1`

Both files source helper scripts using relative paths. Update:

**Before (line 7-8 in both files):**
```powershell
. "$SCRIPT_DIR/run-external-command.ps1"
. "$SCRIPT_DIR/setup_msvc_env.ps1"
```

**After:**
```powershell
. "$SCRIPT_DIR/run-external-command.ps1"
. "$SCRIPT_DIR/setup_msvc_env.ps1"
```

**Note:** These stay the same because all files are in scripts/build/ now!

**Verify:**
```bash
grep 'SCRIPT_DIR' scripts/build/build_and_test.ps1
grep 'SCRIPT_DIR' scripts/build/build_llvm.ps1
```

Paths should still be relative (no changes needed)

### Step 6.6: Update CLAUDE.md

**File:** `CLAUDE.md`

Search for any references to moved scripts and update paths.

**Check for references:**
```bash
grep -n "tools/" CLAUDE.md
grep -n "graph-opt\|onnx-grep\|pattern-gen\|tar" CLAUDE.md
```

**Likely changes:**

If CLAUDE.md mentions tool locations, update from:
```
tools/ directory AND separate tool dirs (graph-opt/, onnx-grep/, etc.)
```

To:
```
tools/ directory (contains graph-opt/, onnx-grep/, pattern-gen/, tar/)
```

**Verify:** Read through CLAUDE.md sections about project structure

---

## Phase 7: Update CMakeLists.txt for Tool Directories

### Step 7.1: Update top-level CMakeLists.txt

**File:** `CMakeLists.txt`

**Before (lines 33-36):**
```cmake
add_subdirectory(graph-opt)
add_subdirectory(tar)
add_subdirectory(pattern-gen)
add_subdirectory(onnx-grep)
```

**After:**
```cmake
add_subdirectory(tools/graph-opt)
add_subdirectory(tools/tar)
add_subdirectory(tools/pattern-gen)
add_subdirectory(tools/onnx-grep)
```

**Verify:**
```bash
grep "add_subdirectory(tools/" CMakeLists.txt
```

Should show 4 lines with tools/ prefix

---

## Phase 8: Test and Verify

### Step 8.1: Test CMake configuration

```bash
# Configure (without building)
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/test-reorganize \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  --fresh
```

**Expected:** No errors about missing directories or files

### Step 8.2: Test Bazel build (if applicable)

```bash
# Try loading the BUILD file
bazel query "//scripts/bazel:all" 2>&1 | head -20
```

**Expected:** patch_wrapper target found

### Step 8.3: Verify file moves

```bash
# Check tools/ only has directories
ls -la tools/
# Should see: graph-opt/, onnx-grep/, pattern-gen/, tar/

# Check scripts structure
ls -la scripts/
# Should see: setup/, build/, debug/, bazel/

# Check cmake structure
ls -la cmake/scripts/
# Should see: xxd.py
```

### Step 8.4: Run pre-commit

```bash
pre-commit run --all-files
```

**Expected:** All checks pass

---

## Phase 9: Commit and Push

### Step 9.1: Stage all changes

```bash
git add -A
```

### Step 9.2: Commit

```bash
git commit -m "docs: add issue #061 - Reorganize tools/ directory"
```

### Step 9.3: Create issue implementation commit

After testing, create implementation commit:

```bash
git commit -m "refactor: reorganize tools/ directory structure

- Move debugging utilities to scripts/debug/
- Move CMake utilities to cmake/scripts/
- Move Bazel utilities to scripts/bazel/
- Move CI build scripts to scripts/build/
- Move tool executables into tools/ (graph-opt, onnx-grep, pattern-gen, tar)
- Delete obsolete files (parse_cl_link_error.py, tools/BUILD.bazel)
- Create scripts/debug/README.md documentation
- Create scripts/bazel/BUILD.bazel for Bazel targets
- Update all references in BUILD.bazel, CI workflows, CMakeLists.txt
- Update CLAUDE.md to reflect new structure

Fixes naming confusion where tools/ contained scripts instead of tools.
Reduces top-level clutter by consolidating 4 scattered tool directories.

Issue #061"
```

### Step 9.4: Push

```bash
git push -u fork feature/exploration-session-059
```

---

## Phase 10: Create Pull Request

```bash
gh pr create --draft --title "Issue #061: refactor: reorganize tools/ directory structure" --body "$(cat <<'EOF'
## Summary
Reorganizes tools/ directory to contain only executable tool components, and organizes scripts into logical categories.

**Changes:**
- ✅ tools/ now contains only executables (graph-opt/, onnx-grep/, pattern-gen/, tar/)
- ✅ Debugging utilities → scripts/debug/ with README.md
- ✅ CMake utilities → cmake/scripts/
- ✅ Bazel utilities → scripts/bazel/ with BUILD.bazel
- ✅ CI build scripts → scripts/build/ (kept for now, future refactoring)
- ✅ Deleted obsolete files (parse_cl_link_error.py, tools/BUILD.bazel)
- ✅ Updated all references (BUILD.bazel, CI workflows, CMakeLists.txt, CLAUDE.md)

**Benefits:**
- Fixes naming confusion (tools/ actually contains tools)
- Reduces top-level clutter (4 scattered dirs → 1)
- Better organization (scripts grouped by purpose)
- Improved developer experience

**Testing:**
- [ ] CMake configuration works
- [ ] Bazel builds work (if tested)
- [ ] Pre-commit passes
- [ ] CI workflows pass

Resolves #061
EOF
)"
```

---

## Success Criteria

- [ ] tools/ contains only 4 directories: graph-opt/, onnx-grep/, pattern-gen/, tar/
- [ ] scripts/debug/ exists with README.md and 2 Python scripts
- [ ] scripts/bazel/ exists with BUILD.bazel, patch_wrapper.py, collect_pb.bzl
- [ ] scripts/build/ exists with 5 PowerShell scripts
- [ ] cmake/scripts/ exists with xxd.py
- [ ] morphizen-core/BUILD.bazel references updated
- [ ] CI workflow files updated (.github/actions/)
- [ ] CMakeLists.txt updated
- [ ] CLAUDE.md updated
- [ ] Pre-commit passes
- [ ] No broken references

---

## Rollback Plan

If issues arise:

```bash
# Revert all changes
git reset --hard origin/main

# Or revert specific commit
git revert <commit-hash>
```

---

## Notes

- This reorganization does NOT modify CI build scripts (build_*.ps1) beyond moving them
- Separate issue will be created for simplifying/refactoring CI build scripts
- This is part of larger top-level directory cleanup effort
