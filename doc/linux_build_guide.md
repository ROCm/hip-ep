

# hipDNNEP and onnx-hipdnn-ep Linux Build Guide

This document provides complete step-by-step instructions for building and testing hipDNNEP and onnx-hipdnn-ep on a Linux test server with AMD ROCm GPU.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Phase 1: Build Original hipDNNEP](#phase-1-build-original-hipdnnep)
4. [Phase 2: Build Morphizen + onnx-hipdnn-ep](#phase-2-build-morphizen--onnx-hipdnn-ep)
5. [Phase 3: Testing and Validation](#phase-3-testing-and-validation)
6. [Environment Variables Reference](#environment-variables-reference)
7. [Troubleshooting](#troubleshooting)
8. [Quick Reference](#quick-reference)

---

## Prerequisites

### System Requirements

- Linux (Ubuntu 20.04 or newer recommended)
- AMD GPU with ROCm support
- Sufficient disk space (at least 50GB recommended)
- Internet connection for downloading dependencies

### Required Software

- CMake 3.28.3 or newer
- Ninja build system 1.11.1 or newer
- Git
- GCC/G++ compiler
- Python 3

### Test Server Details

If using the AMD internal test server:

- **Jump Host**: `xbjjmphost02`
- **Test Server**: `10.176.179.150`
- **Username**: `rocm` (or your assigned username)
- **Password**: `rocm` (or your assigned password)

**GPU Information** (example from test server):
```
Agent 2
  Name:                    gfx1201
  Uuid:                    GPU-5f42131372ae20e9
  Marketing Name:          AMD Radeon RX 9070 XT
  Vendor Name:             AMD
  Chip ID:                 30032(0x7550)
  Cache Info:
    L1:                    32 KB
    L2:                    8192 KB
    L3:                    65536 KB
```

---

## Environment Setup

### 1.1 SSH Connection (for test server)

```bash
# Step 1: Connect to jump host
ssh xbjjmphost02

# Step 2: Connect to ROCm test server from jump host
ssh rocm@10.176.179.150
# Or use your username:
ssh chunywan@10.176.179.150
```

### 1.2 Verify GPU Environment

```bash
# Check GPU information and ROCm version
rocminfo
```

This should display your GPU architecture (e.g., gfx1201).

### 1.3 Verify Build Tools

```bash
# Verify CMake version
cmake --version
# Expected: CMake version 3.28.3 or newer

# Verify Ninja version
ninja --version
# Expected: 1.11.1 or newer

# Verify Git
git --version
```

### 1.4 Set Up Workspace Directory

For this guide, we'll use a workspace directory. Adjust paths as needed for your username.

```bash
# Example for user 'chunywan'
export WORKSPACE_DIR=$HOME/workspace
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Or specify a custom location:
# export WORKSPACE_DIR=/home/chunywan/workspace
# mkdir -p $WORKSPACE_DIR
# cd $WORKSPACE_DIR
```

---

## Phase 1: Build Original hipDNNEP

> **Note: This phase is OPTIONAL**
> 
> Phase 1 builds the original/upstream hipDNNEP to generate reference results for validation purposes.
> 
> **Skip Phase 1 if:**
> - You only need the onnx-hipdnn-ep integration working (proceed to Phase 2)
> - You trust the integration is correct and don't need validation
> - You're time-constrained and just want the integrated version
> 
> **Complete Phase 1 if:**
> - You're developing or debugging the onnx-hipdnn-ep integration
> - You need to verify the integration produces correct results
> - You want to run comparison tests between implementations (Phase 3, section 4.2)
> - You're investigating a bug and need baseline results from the original implementation

### 2.1 Install GTest

```bash
sudo apt update
sudo apt install -y libgtest-dev
```

### 2.2 Install TheRock (ROCm SDK)

TheRock provides HIP, hipDNN, and other ROCm components in a single tarball.

#### Download TheRock Tarball

```bash
cd $WORKSPACE_DIR
mkdir -p therock && cd therock

# Download the latest nightly tarball
# Check https://therock-nightly-tarball.s3.amazonaws.com/ for available versions
# Example for gfx120X (adjust based on your GPU architecture):
wget https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx120X-all-7.11.0a20260106.tar.gz
```

**Note**: Match the tarball to your GPU architecture:
- gfx120X-all for gfx1200, gfx1201
- gfx110X-all for gfx1100, gfx1101, gfx1102, gfx1103
- gfx90X-all for gfx900, gfx906, gfx908, gfx90a, gfx90c

#### Extract TheRock

```bash
# Create installation directory
mkdir -p install

# Extract tarball
tar -xzf therock-dist-linux-gfx120X-all-7.11.0a20260106.tar.gz -C install
```

After extraction, verify:
```bash
ls install/bin/hipconfig
ls install/lib/cmake/hip/hip-config.cmake
```

#### Set TheRock Environment Variables

```bash
export THEROCK_DIST=$WORKSPACE_DIR/therock/install
export PATH=$THEROCK_DIST/bin:$PATH
export HIP_PLATFORM=amd
```

Verify:
```bash
which hipconfig
hipconfig --version
```

### 2.3 Build ONNXRuntime

ONNXRuntime is required for hipDNNEP execution provider interface.

#### Clone ONNXRuntime

```bash
cd $WORKSPACE_DIR
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNXRuntime

```bash
# Build in RelWithDebInfo configuration
./build.sh --config RelWithDebInfo \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync
```

This will create the build output at `build/Linux/RelWithDebInfo/`.

#### Set ONNXRuntime Environment Variable

```bash
export ONNXRUNTIME_ROOT=$WORKSPACE_DIR/onnxruntime
```

### 2.4 Install IREE

IREE compiler is required by hipDNN backend for code generation.

```bash
cd $WORKSPACE_DIR
mkdir -p iree && cd iree

# Download IREE precompiled package
wget https://github.com/iree-org/iree/releases/download/iree-3.10.0rc20260106/iree-dist-3.10.0rc20260106-linux-x86_64.tar.xz

# Extract
tar -xJf iree-dist-3.10.0rc20260106-linux-x86_64.tar.xz
```

#### Add IREE to PATH

```bash
export PATH=$WORKSPACE_DIR/iree/bin:$PATH
```

Verify:
```bash
which iree-compile
iree-compile --version
```

### 2.5 Clone hipDNNEP Repository

Clone hipDNNEP at the specific commit mentioned.

```bash
cd $WORKSPACE_DIR
git clone https://github.com/MaheshRavishankar/hipDNNEP.git
cd hipDNNEP

# Checkout specific commit
git checkout de7921872f218a75e3f6de589a8ed4be9f08782
```

### 2.6 Apply Required Patches to hipDNNEP

#### Patch 1: Remove Hardcoded Paths from CMakePresets.json

The original `CMakePresets.json` contains hardcoded paths from the author's environment. These need to be removed.

```bash
cd $WORKSPACE_DIR/hipDNNEP

# Edit CMakePresets.json
vim CMakePresets.json
```

**Remove these lines** from the `"cacheVariables"` section:
```json
"THEROCK_DIST": "/home/mahesh/TheRock/build/MaheshRelWithDebInfo/dist/rocm",
"ONNXRUNTIME_ROOT": "/home/mahesh/onnxruntime/onnxruntime",
```

**After editing**, the cacheVariables section should look like:
```json
"cacheVariables": {
  "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
  "HIP_PLATFORM": "amd"
}
```

Or apply this patch directly:
```bash
cat > /tmp/cmakepresets.patch << 'EOF'
--- a/CMakePresets.json
+++ b/CMakePresets.json
@@ -14,8 +14,6 @@
       "installDir": "${sourceDir}/../install/hipDNNEP/${presetName}",
       "cacheVariables": {
         "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
-        "THEROCK_DIST": "/home/mahesh/TheRock/build/MaheshRelWithDebInfo/dist/rocm",
-        "ONNXRUNTIME_ROOT": "/home/mahesh/onnxruntime/onnxruntime",
         "HIP_PLATFORM": "amd"
       }
     },
EOF

git apply /tmp/cmakepresets.patch
```

#### Patch 2: Fix ONNXRuntime Library Path in test/CMakeLists.txt

The test configuration looks for ONNXRuntime library at an incorrect path.

```bash
vim test/CMakeLists.txt
```

**Change line 18** from:
```cmake
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
```

**To**:
```cmake
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
```

Or apply this patch:
```bash
cat > /tmp/test_cmake.patch << 'EOF'
--- a/test/CMakeLists.txt
+++ b/test/CMakeLists.txt
@@ -15,7 +15,7 @@ if(NOT GTest_FOUND)
 endif()
 
 # ONNXRuntime library for testing
-set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
+set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
EOF

git apply /tmp/test_cmake.patch
```

### 2.7 Configure hipDNNEP

Ensure environment variables are set:
```bash
# Verify environment variables
echo "THEROCK_DIST=$THEROCK_DIST"
echo "ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
echo "PATH includes iree-compile: $(which iree-compile)"
```

Configure the project:
```bash
cd $WORKSPACE_DIR/hipDNNEP
cmake --preset RelWithDebInfo
```

**Expected output**:
- CMake should find `hip-config.cmake` from TheRock
- CMake should find `hipdnn_frontend` and `hipdnn_backend` from TheRock
- CMake should find `iree-compile` in PATH
- CMake should find ONNXRuntime headers

### 2.8 Build hipDNNEP

```bash
cmake --build --preset RelWithDebInfo
```

**Expected output**:
- Should compile 12 targets (including tests)
- Final output: `[12/12] Linking CXX executable test/hipdnn_ep_tests`

If you see `[9/9]` instead of `[12/12]`, the tests were not built due to ONNXRuntime path issue (check Patch 2 above).

### 2.9 Run hipDNNEP Tests

```bash
cd $WORKSPACE_DIR/hipDNNEP
ctest --preset RelWithDebInfo --output-on-failure
```

**Expected output**:
```
Test project /home/chunywan/workspace/build/hipDNNEP/RelWithDebInfo
    Start 1: HipDNNEpLoadTest.RegisterEpLibrary
1/4 Test #1: HipDNNEpLoadTest.RegisterEpLibrary ........   Passed    0.07 sec
    Start 2: HipDNNEpLoadTest.GetEpDevices
2/4 Test #2: HipDNNEpLoadTest.GetEpDevices .............   Passed    0.04 sec
    Start 3: HipDNNConvTest.BasicConv2D
3/4 Test #3: HipDNNConvTest.BasicConv2D ................   Passed    3.45 sec
    Start 4: HipDNNConvTest.ReferenceConvCorrectness
4/4 Test #4: HipDNNConvTest.ReferenceConvCorrectness ...   Passed    0.04 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   3.60 sec
```

✅ **Phase 1 Complete**: Original hipDNNEP is now built and tested successfully.

---

## Phase 2: Build Morphizen + onnx-hipdnn-ep

### 3.1 Clone Morphizen

```bash
cd $WORKSPACE_DIR
git clone https://github.com/Xilinx/MorphiZen.git --recursive
```

**Note**: The `--recursive` flag is important to clone all submodules.

### 3.2 Clone onnx-hipdnn-ep

```bash
cd $WORKSPACE_DIR
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
```

**Alternative** if using the MaheshRavishankar fork:
```bash
cd $WORKSPACE_DIR/onnx-hipdnn-ep/external
# The hipDNNEP submodule should already point to the correct repository
# Update to specific commit if needed:
cd hipDNNEP
git checkout de7921872f218a75e3f6de589a8ed4be9f08782
cd ../..
```

### 3.3 Build ONNXRuntime with Vitis AI Support (for onnx-hipdnn-ep)

onnx-hipdnn-ep requires ONNXRuntime built with Vitis AI support.

```bash
cd $WORKSPACE_DIR/onnxruntime

# Build with Vitis AI support
./build.sh \
  --config Debug \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --build_dir $WORKSPACE_DIR/build/onnxruntime \
  --skip_tests \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=$WORKSPACE_DIR/local

# Install
cmake --build $WORKSPACE_DIR/build/onnxruntime/Debug/ --target install
```

This installs ONNXRuntime to `$WORKSPACE_DIR/local`.

### 3.4 Configure onnx-hipdnn-ep

```bash
cd $WORKSPACE_DIR/onnx-hipdnn-ep

# Configure with CMake
cmake -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_SHARED_LIBS=OFF \
  -B $WORKSPACE_DIR/build/onnx-hipdnn-ep \
  -S . \
  -DCMAKE_INSTALL_PREFIX=$WORKSPACE_DIR/local \
  -DCMAKE_PREFIX_PATH="$THEROCK_DIST;$WORKSPACE_DIR/local"
```

**Important CMake options**:
- `-DCMAKE_PREFIX_PATH`: Points to both TheRock (for hipDNN) and local install (for ONNXRuntime)
- `-DCMAKE_INSTALL_PREFIX`: Where to install the built libraries
- `-DBUILD_SHARED_LIBS=OFF`: Build static libraries

### 3.5 Build onnx-hipdnn-ep

```bash
cmake --build $WORKSPACE_DIR/build/onnx-hipdnn-ep --config Debug --target install
```

This will:
1. Build the level-1-pass-hipdnn (graph optimization pass)
2. Build custom-op-hipdnn (custom operators)
3. Build proto definitions
4. Build tests
5. Install to `$WORKSPACE_DIR/local`

### 3.6 Verify onnx-hipdnn-ep Build

Check that the key components are built:

```bash
# Check level-1 pass library
ls -lh $WORKSPACE_DIR/build/onnx-hipdnn-ep/level-1-pass-hipdnn/libvitis_ai_ep_level_1.so

# Check custom op library
ls -lh $WORKSPACE_DIR/build/onnx-hipdnn-ep/custom-op-hipdnn/libcustom_op_hipdnn.so

# Check test executable
ls -lh $WORKSPACE_DIR/build/onnx-hipdnn-ep/test/test_onnx_runner
```

---

## Phase 3: Testing and Validation

### 4.1 Run onnx-hipdnn-ep Tests

```bash
cd $WORKSPACE_DIR/build/onnx-hipdnn-ep
ctest --output-on-failure
```

### 4.2 Test Comparison: hipDNNEP vs onnx-hipdnn-ep

#### Test 1: EP Registration (hipDNNEP)

```bash
cd $WORKSPACE_DIR/hipDNNEP
ctest --preset RelWithDebInfo -R RegisterEpLibrary --output-on-failure
```

Expected: Test should pass, confirming EP can be registered.

#### Test 2: Conv2D Operations (hipDNNEP)

```bash
cd $WORKSPACE_DIR/hipDNNEP
ctest --preset RelWithDebInfo -R BasicConv2D --output-on-failure
```

Expected: Test should pass with Conv2D executing successfully on GPU.

#### Test 3: onnx-hipdnn-ep Integration Test

```bash
cd $WORKSPACE_DIR/build/onnx-hipdnn-ep
./test/test_onnx_runner --help
```

Run a Conv2D test through onnx-hipdnn-ep to verify integration:

```bash
# Set environment for debugging
export MORPHIZEN_DEBUG_HIPDNN=1

# Run test (adjust based on available test models)
./test/test_onnx_runner <path_to_test_model.onnx>
```

### 4.3 Validation Checklist

Verify the following functionality works in both hipDNNEP and onnx-hipdnn-ep:

- [ ] EP library loads successfully
- [ ] GPU devices are detected correctly
- [ ] Conv2D operations execute on GPU
- [ ] Reference model produces correct results
- [ ] No memory leaks or crashes
- [ ] Performance is comparable

### 4.4 Key Differences to Note

Document any differences observed between original hipDNNEP and onnx-hipdnn-ep integration:

1. **API differences**: Note any API changes in the integration layer
2. **Performance**: Compare execution times for same operations
3. **Error handling**: Note any differences in error messages or handling
4. **Dependencies**: List additional dependencies required by onnx-hipdnn-ep

---

## Environment Variables Reference

### Required Environment Variables

```bash
# TheRock SDK location
export THEROCK_DIST=$WORKSPACE_DIR/therock/install

# Add TheRock binaries to PATH
export PATH=$THEROCK_DIST/bin:$PATH

# HIP platform
export HIP_PLATFORM=amd

# ONNXRuntime source location
export ONNXRUNTIME_ROOT=$WORKSPACE_DIR/onnxruntime

# IREE compiler
export PATH=$WORKSPACE_DIR/iree/bin:$PATH
```

### Optional Debug Variables

```bash
# Enable hipDNN debugging
export MORPHIZEN_DEBUG_HIPDNN=1
```

### Environment Setup Script

Create a file `$WORKSPACE_DIR/setup_env.sh`:

```bash
#!/bin/bash
# Environment setup for hipDNNEP and onnx-hipdnn-ep development

# Base workspace directory
export WORKSPACE_DIR=$HOME/workspace

# TheRock SDK
export THEROCK_DIST=$WORKSPACE_DIR/therock/install
export PATH=$THEROCK_DIST/bin:$PATH
export HIP_PLATFORM=amd

# ONNXRuntime
export ONNXRUNTIME_ROOT=$WORKSPACE_DIR/onnxruntime

# IREE
export PATH=$WORKSPACE_DIR/iree/bin:$PATH

# Optional: Debug logging
# export MORPHIZEN_DEBUG_HIPDNN=1

echo "Environment configured:"
echo "  WORKSPACE_DIR=$WORKSPACE_DIR"
echo "  THEROCK_DIST=$THEROCK_DIST"
echo "  ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
echo "  iree-compile: $(which iree-compile)"
echo "  hipconfig: $(which hipconfig)"
```

Use it:
```bash
source $WORKSPACE_DIR/setup_env.sh
```

---

## Troubleshooting

### Issue 1: Missing GTest

**Error**:
```
cannot find gtest
```

**Solution**:
```bash
sudo apt update
sudo apt install -y libgtest-dev
```

---

### Issue 2: hip-config.cmake Not Found

**Error**:
```
CMake Error at CMakeLists.txt:27 (find_package):
  Could not find a package configuration file provided by "hip" with any of
  the following names:
    hipConfig.cmake
    hip-config.cmake
```

**Cause**: CMake cannot find HIP configuration files from TheRock.

**Solution**:
```bash
# Verify TheRock is extracted correctly
ls $THEROCK_DIST/lib/cmake/hip/hip-config.cmake

# Ensure TheRock bin is in PATH
export PATH=$THEROCK_DIST/bin:$PATH

# Verify hipconfig works
which hipconfig
hipconfig --version
```

---

### Issue 3: iree-compile Not Found

**Error**:
```
CMake Error at CMakeLists.txt:36 (message):
  iree-compile not found in PATH. It is required by hipDNN backend for code
  generation.
```

**Cause**: IREE compiler is not in PATH.

**Solution**:
```bash
# Add IREE bin directory to PATH
export PATH=$WORKSPACE_DIR/iree/bin:$PATH

# Verify
which iree-compile
iree-compile --version
```

---

### Issue 4: Hardcoded ONNXRuntime Path

**Error**:
```
CMake Error at CMakeLists.txt:50 (message):
  ONNXRuntime headers not found at
  /home/mahesh/onnxruntime/onnxruntime/include/onnxruntime/core/session
```

**Cause**: CMakePresets.json contains hardcoded paths.

**Solution**:
Apply Patch 1 from section 2.6 to remove hardcoded paths from `CMakePresets.json`.

---

### Issue 5: hipdnn_frontend Contains Non-existent Path

**Error**:
```
CMake Error in CMakeLists.txt:
  Imported target "hipdnn_frontend" includes non-existent path
    "/therock/output/build/third-party/flatbuffers/dist/include"
  in its INTERFACE_INCLUDE_DIRECTORIES.
```

**Cause**: TheRock tarball's CMake configuration contains hardcoded build-time paths.

**Solution**:
Manually edit the TheRock CMake configuration files to remove invalid paths:

```bash
cd $THEROCK_DIST/lib/cmake

# Find files with invalid paths
grep -rn "/therock/output/build/third-party/flatbuffers/dist/include" .
grep -rn "/therock/output/build/core/clr/dist/include" .
```

Edit the following files to remove the invalid `INTERFACE_INCLUDE_DIRECTORIES` entries:
- `hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake` (around lines 63-64)
- `hipdnn_backend/hipdnn_backendConfig.cmake` (around lines 63-64)

**Example fix** for `hipdnn_data_sdkTargets.cmake`:
```cmake
# Before (lines 63-64):
INTERFACE_INCLUDE_DIRECTORIES "/therock/output/build/third-party/flatbuffers/dist/include;/opt/rocm/include"

# After:
INTERFACE_INCLUDE_DIRECTORIES "/opt/rocm/include"
```

---

### Issue 6: Tests Not Building

**Warning**:
```
CMake Warning at test/CMakeLists.txt:25 (message):
  ONNXRuntime library not found at
  /home/chunywan/workspace/onnxruntime/build/RelWithDebInfo. Tests will not
  be built.
```

**Cause**: ONNXRuntime library path mismatch (build output is in `build/Linux/RelWithDebInfo`).

**Solution**:
Apply Patch 2 from section 2.6 to fix the path in `test/CMakeLists.txt`.

After applying the fix:
```bash
# Clean and rebuild
rm -rf $WORKSPACE_DIR/build/hipDNNEP
cd $WORKSPACE_DIR/hipDNNEP
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo
```

You should now see `[12/12]` instead of `[9/9]` in the build output.

---

### Issue 7: GPU Not Detected

**Error**:
```
No GPU devices found
```

**Solution**:
```bash
# Verify GPU is visible to ROCm
rocminfo

# Check HIP platform is set
echo $HIP_PLATFORM  # Should output: amd

# Verify HIP installation
hipconfig --version
```

---

### Issue 8: Permission Denied Errors

**Error**:
```
Permission denied when creating directories or files
```

**Solution**:
```bash
# Ensure workspace directory is writable
ls -ld $WORKSPACE_DIR

# Fix permissions if needed
chmod 755 $WORKSPACE_DIR
```

---

## Quick Reference

### Complete Command Sequence

This section provides all commands in sequence for quick copy-paste execution.

```bash
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

export WORKSPACE_DIR=$HOME/workspace
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

# Install GTest
sudo apt update
sudo apt install -y libgtest-dev

# Download and extract TheRock
mkdir -p therock && cd therock
wget https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx120X-all-7.11.0a20260106.tar.gz
mkdir -p install
tar -xzf therock-dist-linux-gfx120X-all-7.11.0a20260106.tar.gz -C install
cd ..

# Download and extract IREE
mkdir -p iree && cd iree
wget https://github.com/iree-org/iree/releases/download/iree-3.10.0rc20260106/iree-dist-3.10.0rc20260106-linux-x86_64.tar.xz
tar -xJf iree-dist-3.10.0rc20260106-linux-x86_64.tar.xz
cd ..

# ============================================================================
# SET ENVIRONMENT VARIABLES
# ============================================================================

export THEROCK_DIST=$WORKSPACE_DIR/therock/install
export PATH=$THEROCK_DIST/bin:$PATH
export HIP_PLATFORM=amd
export PATH=$WORKSPACE_DIR/iree/bin:$PATH

# ============================================================================
# BUILD ONNXRUNTIME
# ============================================================================

cd $WORKSPACE_DIR
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync

export ONNXRUNTIME_ROOT=$WORKSPACE_DIR/onnxruntime

# ============================================================================
# BUILD HIPDNNEP
# ============================================================================

cd $WORKSPACE_DIR
git clone https://github.com/MaheshRavishankar/hipDNNEP.git
cd hipDNNEP
git checkout de7921872f218a75e3f6de589a8ed4be9f08782

# Apply patches (manually edit or use git apply as shown in section 2.6)
# Patch 1: CMakePresets.json - remove hardcoded paths
# Patch 2: test/CMakeLists.txt - fix ONNXRuntime path

# Configure and build
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo

# Run tests
ctest --preset RelWithDebInfo --output-on-failure

# ============================================================================
# BUILD MORPHIZEN + ONNX-HIPDNN-EP
# ============================================================================

# Build ONNXRuntime with Vitis AI support
cd $WORKSPACE_DIR/onnxruntime
./build.sh --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir $WORKSPACE_DIR/build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$WORKSPACE_DIR/local
cmake --build $WORKSPACE_DIR/build/onnxruntime/Debug/ --target install

# Clone repositories
cd $WORKSPACE_DIR
git clone https://github.com/ROCm/MorphiZen.git --recursive
git clone https://github.com/ROCm/onnx-hipdnn-ep.git

# Configure and build onnx-hipdnn-ep
cd onnx-hipdnn-ep
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -B $WORKSPACE_DIR/build/onnx-hipdnn-ep -S . -DCMAKE_INSTALL_PREFIX=$WORKSPACE_DIR/local -DCMAKE_PREFIX_PATH="$THEROCK_DIST;$WORKSPACE_DIR/local"
cmake --build $WORKSPACE_DIR/build/onnx-hipdnn-ep --config Debug --target install

# Run tests
cd $WORKSPACE_DIR/build/onnx-hipdnn-ep
ctest --output-on-failure
```

---

## Directory Structure

After completing all builds, your directory structure should look like:

```
$WORKSPACE_DIR/
├── therock/
│   ├── therock-dist-linux-gfx120X-all-7.11.0a20260106.tar.gz
│   └── install/              # TheRock installation
│       ├── bin/
│       │   ├── hipconfig
│       │   └── ...
│       ├── lib/
│       │   └── cmake/
│       │       ├── hip/
│       │       ├── hipdnn_frontend/
│       │       └── hipdnn_backend/
│       └── include/
├── iree/
│   ├── iree-dist-3.10.0rc20260106-linux-x86_64.tar.xz
│   └── bin/
│       └── iree-compile
├── onnxruntime/              # ONNXRuntime source
│   ├── build/
│   │   └── Linux/
│   │       └── RelWithDebInfo/
│   └── include/
├── hipDNNEP/                 # Original hipDNNEP
│   ├── src/
│   ├── test/
│   └── ...
├── MorphiZen/                # Morphizen framework
│   └── ...
├── onnx-hipdnn-ep/           # onnx-hipdnn-ep integration
│   ├── level-1-pass-hipdnn/
│   ├── custom-op-hipdnn/
│   ├── external/
│   │   └── hipDNNEP/        # hipDNNEP as submodule
│   └── ...
├── build/
│   ├── hipDNNEP/
│   │   └── RelWithDebInfo/
│   ├── onnxruntime/
│   │   └── Debug/
│   └── onnx-hipdnn-ep/
│       └── Debug/
└── local/                    # Installation directory
    ├── bin/
    ├── lib/
    └── include/
```

---

## References

- [hipDNNEP GitHub](https://github.com/MaheshRavishankar/hipDNNEP)
- [TheRock Nightly Tarballs](https://therock-nightly-tarball.s3.amazonaws.com/)
- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [ONNXRuntime GitHub](https://github.com/microsoft/onnxruntime)
- [IREE Releases](https://github.com/iree-org/iree/releases)
- [onnx-hipdnn-ep Documentation](../README.md)

---

## Summary

This guide provides complete instructions for:

1. ✅ Building original hipDNNEP from source with all dependencies
2. ✅ Building onnx-hipdnn-ep with Morphizen framework integration
3. ✅ Testing both implementations to verify functionality
4. ✅ Comparing results to ensure compatibility

### Key Takeaways

- **TheRock** provides all ROCm components including HIP and hipDNN
- **Environment variables** must be set correctly for all builds
- **Patches** are required to remove hardcoded paths from hipDNNEP
- **Tests** verify that both implementations work correctly
- **onnx-hipdnn-ep** integrates hipDNNEP as an execution provider for ONNX models

### Next Steps

After completing this guide:
1. Run performance benchmarks comparing hipDNNEP and onnx-hipdnn-ep
2. Test with various ONNX models to verify graph optimization passes
3. Document any additional integration requirements for production use

---

---

## Windows Build Reference

For Windows builds, see [Windows Build Guide](windows_build_guide.md).

### Sample Windows Build Output

#### CMake Configuration (Windows)

```
PS C:\Develop\m\build\hipDNNEP> cmake "C:\Develop\m\Source\onnx-hipdnn-ep\external\hipDNNEP" -G Ninja ...

-- The CXX compiler identification is Clang 21.1.8 with GNU-like command-line
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/LLVM/bin/clang++.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test HIP_CLANG_SUPPORTS_PARALLEL_JOBS
-- Performing Test HIP_CLANG_SUPPORTS_PARALLEL_JOBS - Failed
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - no
-- Found Threads: TRUE
-- Found nlohmann_json: C:/Develop/m/dist/therock/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake (found version "3.12.0")
-- hipDNN Data SDK: Engine plugin build directory is /hipdnn_plugins/engines
-- hipDNN Data SDK: Plugin absolute installation directory C:/Develop/m/dist/therock/bin/hipdnn_plugins/engines
-- hipDNN Data SDK: Plugin relative installation directory bin/hipdnn_plugins/engines
-- Found iree-compile: C:/Users/amd/AppData/Local/Programs/Python/Python312/Scripts/iree-compile.exe
-- Configuring done (2.1s)
-- Generating done (0.1s)
-- Build files have been written to: C:/Develop/m/build/hipDNNEP
```

#### Build Output (Windows)

```
PS C:\Develop\m\build\hipDNNEP> ninja

[1/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/ep_utils.cc.obj
[2/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/hipdnn_ep_exports.cc.obj
[3/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/ep_data_transfer.cc.obj
[4/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/ep_allocator.cc.obj
[5/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/ep_factory.cc.obj
[6/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/node_compute_info.cc.obj
[7/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/ep.cc.obj
[8/9] Building CXX object CMakeFiles/hipdnn_ep.dir/src/kernel.cc.obj
[9/9] Linking CXX shared library hipdnn_ep.dll
```

#### Build Artifacts (Windows)

```
PS C:\Develop\m\build\hipDNNEP> dir *.dll, *.lib

Name          Length LastWriteTime
----          ------ -------------
hipdnn_ep.dll 629760 1/11/2026 10:27:36 PM
hipdnn_ep.lib   1604 1/11/2026 10:27:36 PM
```

### Windows Prerequisites

| Tool | Version | Installation |
|------|---------|--------------|
| CMake | 4.2.1+ | `winget install Kitware.CMake` |
| Ninja | 1.13.2+ | `winget install Ninja-build.Ninja` |
| Clang/LLVM | 21.1.8+ | `winget install LLVM.LLVM` |
| Python | 3.12+ | `winget install Python.Python.3.12` |
| IREE | 3.9.0+ | `pip install iree-base-compiler` |

### Windows Environment Variables

```powershell
$env:THEROCK_PATH = "C:\Develop\m\dist\therock"
$env:ONNXRUNTIME_ROOT = "C:\Develop\m\dist\onnxruntime\onnxruntime-win-x64-1.21.0"
$env:HIP_PLATFORM = "amd"
$env:PATH = "C:\Program Files\LLVM\bin;$env:THEROCK_PATH\bin;$env:PATH"
```

### Windows-Specific Fixes Required

1. **TheRock Hardcoded Paths**: Replace `B:/build/*` paths in cmake configs:
   ```powershell
   # Fix paths in TheRock cmake configs
   (Get-Content "$env:THEROCK_PATH\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake") `
     -replace "B:/build/third-party/flatbuffers/dist/include", "$env:THEROCK_PATH/include" `
     | Set-Content "$env:THEROCK_PATH\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake"
   ```

2. **ONNX Runtime EP Headers**: Copy EP API headers from source:
   ```powershell
   # Clone ORT source and copy EP headers
   git clone --depth 1 https://github.com/microsoft/onnxruntime.git C:\Develop\m\source\onnxruntime
   Copy-Item "C:\Develop\m\source\onnxruntime\include\onnxruntime\core\session\*.h" `
     -Destination "$env:ONNXRUNTIME_ROOT\include\onnxruntime\core\session\" -Force
   ```

---

**Last Updated**: 2026-01-11  
**Build Environment**: Linux with AMD ROCm GPU, Windows 11 with AMD GPU  
**Tested On**: AMD Radeon RX 9070 XT (gfx1201), AMD Radeon (gfx1100)  
**Build Tools**: CMake 3.28.3+/4.2.1, Ninja 1.11.1+/1.13.2, Clang 21.1.8
