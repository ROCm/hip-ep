# onnx-hipdnn-ep Linux Build Guide

This document provides complete step-by-step instructions for building and testing onnx-hipdnn-ep on a Linux test server with AMD ROCm GPU.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Phase 1: Build ONNXRuntime](#phase-1-build-onnxruntime)
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

## Phase 1: Install TheRock and Build ONNXRuntime

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

#### Clone ONNXRuntime

```bash
cd $WORKSPACE_DIR
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNXRuntime

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

### 3.3 Configure onnx-hipdnn-ep

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

### 3.4 Build onnx-hipdnn-ep

```bash
cmake --build $WORKSPACE_DIR/build/onnx-hipdnn-ep --config Debug --target install
```

This will:
1. Build the level-1-pass-hipdnn (graph optimization pass)
2. Build custom-op-hipdnn (custom operators)
3. Build proto definitions
4. Build tests
5. Install to `$WORKSPACE_DIR/local`

### 3.5 Verify onnx-hipdnn-ep Build

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

### 4.2 Validation Checklist

Verify the following functionality works in  onnx-hipdnn-ep:

- [ ] EP library loads successfully
- [ ] GPU devices are detected correctly
- [ ] Conv2D operations execute on GPU
- [ ] Reference model produces correct results
- [ ] No memory leaks or crashes
- [ ] Performance is comparable

### 4.4 Key Differences to Note

Document any differences observed onnx-hipdnn-ep integration:

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
# Environment setup for onnx-hipdnn-ep development

# Base workspace directory
export WORKSPACE_DIR=$HOME/workspace

# TheRock SDK
export THEROCK_DIST=$WORKSPACE_DIR/therock/install
export PATH=$THEROCK_DIST/bin:$PATH
export HIP_PLATFORM=amd

# ONNXRuntime
export ONNXRUNTIME_ROOT=$WORKSPACE_DIR/onnxruntime


# Optional: Debug logging
# export MORPHIZEN_DEBUG_HIPDNN=1

echo "Environment configured:"
echo "  WORKSPACE_DIR=$WORKSPACE_DIR"
echo "  THEROCK_DIST=$THEROCK_DIST"
echo "  ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
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


### Issue 3: hipdnn_frontend Contains Non-existent Path

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


### Issue 4: GPU Not Detected

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

### Issue 5: Permission Denied Errors

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


# ============================================================================
# SET ENVIRONMENT VARIABLES
# ============================================================================

export THEROCK_DIST=$WORKSPACE_DIR/therock/install
export PATH=$THEROCK_DIST/bin:$PATH
export HIP_PLATFORM=amd

# ============================================================================
# BUILD ONNXRUNTIME
# ============================================================================

# Build ONNXRuntime with Vitis AI support
cd $WORKSPACE_DIR/onnxruntime
./build.sh --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir $WORKSPACE_DIR/build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$WORKSPACE_DIR/local
cmake --build $WORKSPACE_DIR/build/onnxruntime/Debug/ --target install


# ============================================================================
# BUILD MORPHIZEN + ONNX-HIPDNN-EP
# ============================================================================

# Clone repositories
cd $WORKSPACE_DIR
git clone ../MorphiZen --recursive
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
├── onnxruntime/              # ONNXRuntime source
│   ├── build/
│   │   └── Linux/
│   │       └── RelWithDebInfo/
│   └── include/
├── MorphiZen/                # Morphizen framework
│   └── ...
├── onnx-hipdnn-ep/           # onnx-hipdnn-ep integration
│   ├── level-1-pass-hipdnn/
│   ├── custom-op-hipdnn/
│   └── ...
├── build/
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

- [TheRock Nightly Tarballs](https://therock-nightly-tarball.s3.amazonaws.com/)
- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [ONNXRuntime GitHub](https://github.com/microsoft/onnxruntime)
- [onnx-hipdnn-ep Documentation](../README.md)

---

## Summary

This guide provides complete instructions for:

1. ✅ Building onnx-hipdnn-ep with Morphizen framework integration
2. ✅ Testing both implementations to verify functionality
3. ✅ Comparing results to ensure compatibility

### Key Takeaways

- **TheRock** provides all ROCm components including HIP and hipDNN
- **Environment variables** must be set correctly for all builds
- **Tests** verify that both implementations work correctly
- **onnx-hipdnn-ep**  an execution provider for ONNX models

---

---

## Windows Build Reference

For Windows builds, see [Windows Build Guide](windows_build_guide.md).

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
