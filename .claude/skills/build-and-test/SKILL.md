<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

---
name: build-and-test
description: Build and test MorphiZen with automated dependency management
allowed-tools: [Bash, Read, Grep]
---

# Build and Test MorphiZen

Automates the build/test workflow with dependency checking and error remediation. See `CLAUDE.md` for build command reference and `docs/developer-guide.md` for manual dependency builds.

## Key Automation

1. **MSVC environment check** (Windows) - verifies `cl.exe` availability
2. **Auto-build missing dependencies** - protobuf, gtest, glog with correct runtime library flags
3. **Build type detection** - auto-chooses clean vs incremental build
4. **Error remediation** - runtime library mismatches, missing deps, test failures

## Instructions for Claude

### Step 0: Git Workflow Check

**Build/test operations** don't modify source → can run on any branch.

**Code changes** (fixing errors) → require feature branch:
- If on `main`: create `feature/<name>` branch first
- Follow CLAUDE.md git workflow (conventional commits, push to fork)

Check branch: `git branch --show-current`

### Step 1: MSVC Check (Windows Only)

```bash
which cl.exe 2>/dev/null
```

If not found, stop and inform user:
```
MSVC compiler not found. Launch git-bash from "Developer Command Prompt for VS 20XX":
1. Open "Developer Command Prompt for VS 20XX"
2. Run: bash
3. Navigate to project and launch Claude Code
```

### Step 2: Auto-Build Missing Dependencies

**Setup paths:**
```bash
WORKSPACE=$(cd ../.. && pwd)
LOCAL_DIR="$WORKSPACE/local"
```

**Check and build in order:**

1. **protobuf v21.12** (build first - others depend on it)
2. **gtest v1.15.0**
3. **glog v0.7.1** (fixes auto-fetch runtime library issues)
4. **ONNX Runtime** - if missing, stop and direct to `docs/developer-guide.md` (complex VitisAI build, cannot auto-build)

**For each dependency** (protobuf, gtest, glog):
```bash
# Check if installed
if ! ls "$LOCAL_DIR/lib/cmake/<dep>/*.cmake" 2>/dev/null; then
  # Clone to ../../../source/<dep>
  # Build to ../../build/<dep>
  # Install to $LOCAL_DIR
  # Use these flags:
  #   -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>
  #   -DCMAKE_BUILD_TYPE=Debug
  #   -DCMAKE_INSTALL_PREFIX="$LOCAL_DIR"
  #   -DBUILD_SHARED_LIBS=OFF
  #   -DBUILD_TESTING=OFF (for glog)
fi
```

**See `docs/developer-guide.md` for:**
- Exact git clone URLs and versions
- Dependency-specific CMake options
- Detailed build commands

**CRITICAL**: All deps must use `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>` to avoid linker errors.

### Step 3: Determine Build Type

Auto-detect based on build directory state:
```bash
# Check if CMakeCache.txt exists
if [ -f ../../build/$(basename $PWD)/CMakeCache.txt ]; then
  # Incremental build
else
  # Clean build
fi
```

Only ask user if previous build failed or user explicitly requests clean build.

### Step 4: Execute Build

**Always compute absolute path for CMAKE_PREFIX_PATH:**
```bash
LOCAL_DIR=$(cd ../../local && pwd)
```

**Clean build:**
```bash
cmake -S . -B ../../build/$(basename $PWD) \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  --fresh

cmake --build ../../build/$(basename $PWD) --config Debug --parallel
ctest --test-dir ../../build/$(basename $PWD) -C Debug --output-on-failure --timeout 600
```

**Incremental build:**
```bash
cmake --build ../../build/$(basename $PWD) --config Debug --parallel
ctest --test-dir ../../build/$(basename $PWD) -C Debug --output-on-failure --timeout 600
```

Parse output for errors and show clear summaries.

### Step 5: Error Remediation

#### Runtime Library Mismatch

**Symptoms:** `LNK2038: mismatch detected for 'RuntimeLibrary'` or `LNK2005: symbol already defined in libcmt.lib`

**Solution:**
1. Identify problematic dependency from linker error
2. Rebuild that dependency with `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`
3. For glog specifically: use `-DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF`

#### Missing ONNX Runtime

**Symptoms:** `Could not find package onnxruntime`

**Solution:** Direct user to `docs/developer-guide.md` - ONNX Runtime requires VitisAI support and manual build.

#### Build Errors

1. Show error with file:line reference
2. Read affected file for context
3. If fixing code: verify feature branch, commit with `fix:` message, push to fork

#### Test Failures

1. Show failed test names and output
2. Suggest individual test run: `ctest --test-dir ../../build/$(basename $PWD) -C Debug -R <test_name> --verbose`

## Known Issues

### Glog Auto-Fetch Runtime Library Mismatch

When CMake auto-fetches glog, it may build glog's tests with `/MDd` instead of `/MTd`, causing linker errors.

**Fix:** Pre-build glog to `../../local`:
```bash
git clone --branch v0.7.1 --depth 1 https://github.com/google/glog.git ../glog
cmake -S ../glog -B ../../build/glog -G Ninja \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_INSTALL_PREFIX=../../local \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF
cmake --build ../../build/glog --config Debug --parallel
cmake --install ../../build/glog --config Debug
```

Verify: `ls ../../local/lib/glogd.lib` (NOT glogd.dll)

### CTest Discovery Error

**Symptom:** `Error loading "onnxruntime_providers_vitisai.dll"` during CMake configure.

**Impact:** Doesn't affect build or test execution. Tests work when run manually.

**Workaround:** Run tests directly or use `ctest --test-dir ../../build/$(basename $PWD) -C Debug --verbose`
