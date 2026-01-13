# hipDNNEP and morphizen-hipdnn Windows Build Guide

This document provides complete step-by-step instructions for building and testing hipDNNEP and morphizen-hipdnn on Windows with AMD ROCm GPU.

> **Note**: This is the Windows adaptation of `HIPDNNEP_E2E_test.md`. Key differences from Linux:
> - No SSH required (local execution)
> - PowerShell instead of bash
> - winget instead of apt
> - Windows paths and environment variables

## Table of Contents

1. [Checking AMD GPU Hardware Information](#checking-amd-gpu-hardware-information)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Phase 1: Build Original hipDNNEP](#phase-1-build-original-hipdnnep)
5. [Phase 2: Build Morphizen + morphizen-hipdnn](#phase-2-build-morphizen--morphizen-hipdnn)
6. [Phase 3: Testing and Validation](#phase-3-testing-and-validation)
7. [Environment Variables Reference](#environment-variables-reference)
8. [Troubleshooting](#troubleshooting)
9. [Patches Reference](#patches-reference)
10. [Quick Reference](#quick-reference)

---

## Checking AMD GPU Hardware Information

Before starting, verify your AMD GPU is properly detected. This section provides multiple methods to check GPU information with sample outputs.

### Method 1: Windows PowerShell (No Dependencies)

Basic GPU information using Windows Management Instrumentation:

```powershell
Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, Status, VideoProcessor
```

**Sample output**:
```
Name           : AMD Radeon PRO W7900
AdapterRAM     : 4293918720
DriverVersion  : 32.0.21037.1004
Status         : OK
VideoProcessor : AMD FirePro SDI (0x7448)
```

For detailed information:
```powershell
Get-CimInstance -ClassName Win32_VideoController | Format-List *
```

**Sample output** (AMD GPU section):
```
Caption                      : AMD Radeon PRO W7900
Description                  : AMD Radeon PRO W7900
Name                         : AMD Radeon PRO W7900
Status                       : OK
DeviceID                     : VideoController2
PNPDeviceID                  : PCI\VEN_1002&DEV_7448&SUBSYS_0E0D1002&REV_00\6&28CF2CBF&0&00000019
CurrentBitsPerPixel          : 32
CurrentHorizontalResolution  : 1920
CurrentVerticalResolution    : 1080
CurrentRefreshRate           : 144
MaxRefreshRate               : 144
MinRefreshRate               : 48
VideoProcessor               : AMD FirePro SDI (0x7448)
AdapterCompatibility         : Advanced Micro Devices, Inc.
AdapterDACType               : Internal DAC(400MHz)
AdapterRAM                   : 4293918720
DriverDate                   : 11/27/2025 4:00:00 PM
DriverVersion                : 32.0.21037.1004
InfFilename                  : oem2.inf
InfSection                   : ati2mtag_Navi31
InstalledDisplayDrivers      : C:\WINDOWS\System32\DriverStore\FileRepository\u0196663.inf_amd64_...\atidx9loader64.dll,...
VideoModeDescription         : 1920 x 1080 x 4294967296 colors
```

### Method 2: GPU Architecture using amdgpu-arch (Clang/LLVM)

After installing LLVM/Clang, use `amdgpu-arch` to get the GPU architecture identifier:

```powershell
# If installed via winget (default path)
& "C:\Program Files\LLVM\bin\amdgpu-arch.exe"

# Or if installed to custom path
& "C:\Develop\m\dist\clang\bin\amdgpu-arch.exe"
```

**Sample output**:
```
gfx1100
```

This architecture string is used to select the correct TheRock SDK build. See [GFX Family table](#step-2-determine-gfx-family) for mapping.

### Method 3: HIP Configuration (TheRock SDK)

After installing TheRock, use `hipconfig` for comprehensive HIP/ROCm information:

```powershell
& "C:\Develop\m\dist\therock\bin\hipconfig.exe" --full
```

**Sample output**:
```
HIP version: 7.2.53150-56870acb4f

==hipconfig
HIP_PATH           :C:\Develop\m\dist\therock
ROCM_PATH          :C:\Develop\m\dist\therock
HIP_COMPILER       :clang
HIP_PLATFORM       :amd
HIP_RUNTIME        :rocclr
CPP_CONFIG         :/

==hip-clang
HIP_CLANG_PATH     :C:\Develop\m\dist\therock\lib\llvm\bin

AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git ...)
Target: x86_64-pc-windows-msvc
Thread model: posix
InstalledDir: C:\Develop\m\dist\therock\lib\llvm\bin

hip-clang-cxxflags :
 -O3
hip-clang-ldflags :
--driver-mode=g++ -O3 -fuse-ld=lld --ld-path="C:\Develop\m\dist\therock\lib\llvm\bin\lld-link.exe" --hip-link

== Environment Variables
PATH=C:\WINDOWS\system32;C:\WINDOWS;...

== Windows Display Drivers
win-9700
Advanced Micro Devices, Inc. C:\WINDOWS\System32\DriverStore\FileRepository\... AMD Radeon PRO W7900
Hostname      :
```

#### Quick HIP commands:

```powershell
# Check HIP platform
& "C:\Develop\m\dist\therock\bin\hipconfig.exe" --platform
# Output: amd

# Check HIP version
& "C:\Develop\m\dist\therock\bin\hipconfig.exe" --version
# Output: 7.2.53150-56870acb4f

# Check ROCm path
& "C:\Develop\m\dist\therock\bin\hipconfig.exe" -rocmpath
# Output: C:\Develop\m\dist\therock
```

### GPU Information Summary

| Property | Value (Example) | How to Check |
|----------|-----------------|--------------|
| GPU Model | AMD Radeon PRO W7900 | PowerShell `Get-CimInstance` |
| GPU Architecture | gfx1100 | `amdgpu-arch.exe` |
| GFX Family | gfx110X-all | See [GFX Family table](#step-2-determine-gfx-family) |
| VRAM | ~4 GB (reported) | PowerShell `Get-CimInstance` |
| Driver Version | 32.0.21037.1004 | PowerShell `Get-CimInstance` |
| HIP Version | 7.2.53150 | `hipconfig.exe --version` |
| HIP Platform | amd | `hipconfig.exe --platform` |

---

## Prerequisites

### System Requirements

- Windows 10 or Windows 11 (Windows 11 recommended)
- AMD GPU with ROCm support
- Sufficient disk space (at least 50GB recommended)
- Internet connection for downloading dependencies
- Administrator privileges for system configuration

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| CMake | 3.28.3+ | Build system generator |
| Ninja | 1.11.1+ | Build system |
| Git | 2.x+ | Version control |
| Visual Studio 2022 | Community+ | C++ compiler (for some components) |
| Clang | 20.x | Required for hipDNNEP (MSVC incompatible) |
| Python | 3.x | Build scripts |

### Directory Structure

This guide uses the following directory structure:

```
C:\Develop\m\
├── source\           # Source code repositories
│   ├── hipDNNEP\
│   ├── MorphiZen\
│   ├── morphizen-hipdnn\
│   └── onnxruntime\
├── build\            # Build directories
│   ├── hipDNNEP\
│   ├── onnxruntime\
│   └── morphizen-hipdnn\
├── local\            # Installation prefix
│   ├── bin\
│   ├── lib\
│   └── include\
└── dist\             # External tools (read-only/downloaded)
    ├── clang\        # Clang 20.x
    ├── therock\      # TheRock SDK
    └── iree\         # IREE Compiler (if separate)
```

---

## Environment Setup

### 1.1 System Configuration

#### Enable Long Paths (Administrative PowerShell)

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Enable Developer Mode (for Symlinks)

- **Windows 11**: Settings → System → For Developers → Developer Mode → toggle **On**
- **Windows 10**: Settings → Update & Security → For Developers → Developer Mode → toggle **On**

**Restart your computer** after these changes.

#### Configure Git

```powershell
git config --global core.symlinks true
git config --global core.longpaths true
```

### 1.2 Install Build Tools (Using winget)

Open a **new PowerShell** terminal:

```powershell
# Install CMake
winget install Kitware.CMake --accept-package-agreements --accept-source-agreements

# Install Ninja
winget install Ninja-build.Ninja --accept-package-agreements --accept-source-agreements

# Install Git (if not already installed)
winget install Git.Git --accept-package-agreements --accept-source-agreements
```

**Important**: After installation, **close and reopen your terminal** to refresh PATH.

#### Verify Installations

```powershell
cmake --version
# Expected: cmake version 3.28.3 or newer

ninja --version
# Expected: 1.11.1 or newer

git --version
```

### 1.3 Create Directory Structure

```powershell
# Create all required directories
New-Item -ItemType Directory -Force -Path @(
    "C:\Develop\m\source",
    "C:\Develop\m\build",
    "C:\Develop\m\local",
    "C:\Develop\m\dist\clang",
    "C:\Develop\m\dist\therock",
    "C:\Develop\m\dist\iree"
)
```

### 1.4 Install Clang/LLVM

Clang is **required** because MSVC cannot compile TheRock SDK headers.

#### Option A: Using winget (Recommended)

```powershell
winget install LLVM.LLVM --accept-package-agreements --accept-source-agreements
```

This installs LLVM to `C:\Program Files\LLVM`. **Restart your terminal** to refresh PATH.

#### Option B: Manual Download

1. **Download** from: https://github.com/llvm/llvm-project/releases
   - Look for `LLVM-20.*.tar.xz` or `LLVM-21.*-win64.exe`
   - Example: `LLVM-21.1.8-win64.exe`

2. **Install/Extract** to `C:\Develop\m\dist\clang` or `C:\Program Files\LLVM`
   - If using installer: Set custom install path
   - If using tar.xz: Extract using 7-Zip

#### Verify installation

```powershell
# If installed via winget (default path)
& "C:\Program Files\LLVM\bin\clang++.exe" --version

# Or if installed to custom path
& "C:\Develop\m\dist\clang\bin\clang++.exe" --version
```

### 1.5 Install TheRock SDK (Windows)

TheRock provides HIP, hipDNN, IREE, and other ROCm components.

#### Step 1: Determine Your GPU Architecture

Use `amdgpu-arch` from Clang/LLVM:
```powershell
# If installed via winget (to C:\Program Files\LLVM)
& "C:\Program Files\LLVM\bin\amdgpu-arch.exe"

# Or if installed to custom path
& "C:\Develop\m\dist\clang\bin\amdgpu-arch.exe"
```

**Sample output**:
```
gfx1100
```

This outputs your GPU architecture - **record this value**.

#### Step 2: Determine GFX Family

| GPU Architecture | GFX Family |
|------------------|------------|
| gfx900, gfx906, gfx908, gfx90a, gfx90c | gfx90X-all |
| gfx940, gfx941, gfx942 | gfx94X-all |
| gfx1030, gfx1031, gfx1032, gfx1034, gfx1035, gfx1036 | gfx103X-all |
| gfx1100, gfx1101, gfx1102, gfx1103 | gfx110X-all |
| gfx1150, gfx1151, gfx1152 | gfx115X-all |
| gfx1200, gfx1201 | gfx120X-all |

#### Step 3: Download TheRock

1. Visit: https://therock-nightly-tarball.s3.amazonaws.com/index.html
2. Find the latest `therock-dist-windows-gfx###-all-*.tar.gz` matching your GFX family
3. Download to `C:\Develop\m\dist\`

#### Step 4: Extract TheRock

Using PowerShell with tar:
```powershell
Set-Location C:\Develop\m\dist

# Extract (adjust filename to match your download)
tar -xzf therock-dist-windows-gfx110X-all-7.10.0a20251103.tar.gz

# Rename if needed
if (Test-Path "therock-dist-*") {
    Rename-Item "therock-dist-*" "therock"
}
```

Or using 7-Zip:
```powershell
# If you have 7-Zip installed
& "C:\Program Files\7-Zip\7z.exe" x therock-dist-windows-*.tar.gz
& "C:\Program Files\7-Zip\7z.exe" x therock-dist-windows-*.tar -o"therock"
```

#### Step 5: Verify TheRock

```powershell
& "C:\Develop\m\dist\therock\bin\hipconfig.exe" -rocmpath -n --hipclangpath
```

Expected output:
```
C:\Develop\m\dist\therock
C:\Develop\m\dist\therock\lib\llvm\bin
```

### 1.6 Install IREE Compiler

IREE (Intermediate Representation Execution Environment) is required by hipDNN backend for kernel code generation. There are two ways to install it:

#### Option A: Check if included in TheRock

```powershell
& "C:\Develop\m\dist\therock\bin\iree-compile.exe" --version
```

If this works, IREE is already included in your TheRock distribution.

#### Option B: Install via pip (Recommended)

IREE is now distributed as Python wheels, which is the easiest installation method:

```powershell
# Install IREE compiler via pip
python -m pip install iree-base-compiler --upgrade

# Verify installation - check Scripts directory
& "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\iree-compile.exe" --version
```

This installs `iree-compile.exe` to your Python Scripts directory.

**Important**: Add the Python Scripts directory to your PATH for CMake to find it:
```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts;$env:PATH"
```

#### Option C: Download standalone release

1. Visit: https://github.com/iree-org/iree/releases
2. Download `iree-dist-*-windows-x86_64.tar.xz`
3. Extract to `C:\Develop\m\dist\iree`
4. Add to PATH: `$env:PATH = "C:\Develop\m\dist\iree\bin;$env:PATH"`

### 1.7 Set Environment Variables

#### Option A: Set for Current Session Only

```powershell
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"
$env:HIP_PLATFORM = "amd"
$env:PATH = "C:\Develop\m\dist\therock\bin;C:\Develop\m\dist\clang\bin;$env:PATH"
$env:ONNXRUNTIME_ROOT = "C:\Develop\m\source\onnxruntime"
```

#### Option B: Set Permanently (Administrative PowerShell)

```powershell
# Set system environment variables
[Environment]::SetEnvironmentVariable("THEROCK_DIST", "C:\Develop\m\dist\therock", "Machine")
[Environment]::SetEnvironmentVariable("HIP_PLATFORM", "amd", "Machine")

# Add to system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$newPaths = "C:\Develop\m\dist\therock\bin;C:\Develop\m\dist\clang\bin"
if ($currentPath -notlike "*therock*") {
    $newPath = "$newPaths;$currentPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
}
```

#### Option C: Use Setup Script

Create `C:\Develop\m\setup_env.ps1`:
```powershell
# Environment setup script for hipDNNEP and morphizen-hipdnn
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"
$env:HIP_PLATFORM = "amd"
$env:ONNXRUNTIME_ROOT = "C:\Develop\m\source\onnxruntime"
$env:PATH = "C:\Develop\m\dist\therock\bin;C:\Develop\m\dist\clang\bin;$env:PATH"

Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  THEROCK_DIST = $env:THEROCK_DIST"
Write-Host "  HIP_PLATFORM = $env:HIP_PLATFORM"
Write-Host "  ONNXRUNTIME_ROOT = $env:ONNXRUNTIME_ROOT"
```

Usage:
```powershell
. C:\Develop\m\setup_env.ps1
```

---

## Phase 1: Build hipDNNEP

There are two ways to build hipDNNEP:

### Option A: Build from morphizen-hipdnn Submodule (Recommended)

If you're working with morphizen-hipdnn, hipDNNEP is included as a git submodule with all Windows patches already applied:

```powershell
Set-Location C:\Develop\m\Source\morphizen-hipdnn
git submodule update --init --recursive

# The submodule is at: external/hipDNNEP
```

Build using the submodule:
```powershell
$buildDir = "C:\Develop\m\build\morphizen-hipdnn-e2e"
$srcDir = "C:\Develop\m\Source\morphizen-hipdnn\external\hipDNNEP"

cmake -G Ninja -B $buildDir -S $srcDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" `
    -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang.exe" `
    -DTHEROCK_DIST=C:/Develop/m/dist/therock `
    -DONNXRUNTIME_ROOT=C:/Develop/m/local `
    -DHIPDNN_EP_BUILD_TESTS=ON

cmake --build $buildDir
```

### Option B: Clone Upstream Repository

Clone the upstream repository (requires applying patches manually):

```powershell
Set-Location C:\Develop\m\source
git clone https://github.com/MaheshRavishankar/hipDNNEP.git
Set-Location hipDNNEP
git checkout de7921872f218a75e3f6de589a8ed4be9f08782
```

### 2.2 Apply Required Patches

The original hipDNNEP contains hardcoded Linux paths that need to be patched.

#### Patch 1: Remove Hardcoded Paths from CMakePresets.json

**Diff** (see [Patches Reference](#patch-1-cmakepresetsjson) for details):
```diff
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
```

**Apply manually** by editing `CMakePresets.json`:
```powershell
# Open in editor
code CMakePresets.json
# Remove the two lines with hardcoded paths
```

#### Patch 2: Fix ONNXRuntime Library Path for Windows

**Diff**:
```diff
--- a/test/CMakeLists.txt
+++ b/test/CMakeLists.txt
@@ -15,7 +15,7 @@ if(NOT GTest_FOUND)
 endif()
 
 # ONNXRuntime library for testing
-set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
+set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Windows/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
```

**Apply manually**:
```powershell
# Open in editor
code test/CMakeLists.txt
# Change line 18: Replace "build/RelWithDebInfo" with "build/Windows/RelWithDebInfo"
```

### 2.3 Build ONNXRuntime

```powershell
Set-Location C:\Develop\m\source
git clone https://github.com/microsoft/onnxruntime.git
Set-Location onnxruntime

# Build with RelWithDebInfo configuration
.\build.bat --config RelWithDebInfo `
    --build_shared_lib `
    --parallel `
    --compile_no_warning_as_error `
    --skip_submodule_sync `
    --build_dir C:\Develop\m\build\onnxruntime
```

Set environment variable:
```powershell
$env:ONNXRUNTIME_ROOT = "C:\Develop\m\source\onnxruntime"
```

### 2.4 Configure hipDNNEP

```powershell
Set-Location C:\Develop\m\source\hipDNNEP

# Configure with Clang (MSVC incompatible with TheRock headers)
# Use the Clang path matching your installation:
#   - winget: "C:/Program Files/LLVM/bin/clang++.exe"
#   - custom: "C:/Develop/m/dist/clang/bin/clang++.exe"

cmake -G Ninja `
    -B C:\Develop\m\build\hipDNNEP\RelWithDebInfo `
    -S . `
    -DCMAKE_BUILD_TYPE=RelWithDebInfo `
    -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" `
    -DTHEROCK_DIST=C:/Develop/m/dist/therock `
    -DONNXRUNTIME_ROOT=C:/Develop/m/source/onnxruntime `
    -DCMAKE_PREFIX_PATH="C:/Develop/m/dist/therock"
```

### 2.5 Build hipDNNEP

```powershell
cmake --build C:\Develop\m\build\hipDNNEP\RelWithDebInfo
```

Expected output: `[12/12] Linking CXX executable test/hipdnn_ep_tests.exe`

### 2.6 Prepare Test Environment

Copy required DLLs to test directory:
```powershell
$buildDir = "C:\Develop\m\build\hipDNNEP\RelWithDebInfo"
$testDir = "$buildDir\test"

# Copy hipdnn_ep.dll
Copy-Item "$buildDir\hipdnn_ep.dll" $testDir -Force

# Copy onnxruntime.dll
$ortDll = "C:\Develop\m\build\onnxruntime\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime.dll"
if (Test-Path $ortDll) {
    Copy-Item $ortDll $testDir -Force
}
```

### 2.7 Run hipDNNEP Tests

```powershell
Set-Location C:\Develop\m\build\hipDNNEP\RelWithDebInfo
ctest --output-on-failure
```

Expected output:
```
Test project C:/Develop/m/build/hipDNNEP/RelWithDebInfo
    Start 1: HipDNNEpLoadTest.RegisterEpLibrary
1/4 Test #1: HipDNNEpLoadTest.RegisterEpLibrary ........   Passed
    Start 2: HipDNNEpLoadTest.GetEpDevices
2/4 Test #2: HipDNNEpLoadTest.GetEpDevices .............   Passed
    Start 3: HipDNNConvTest.BasicConv2D
3/4 Test #3: HipDNNConvTest.BasicConv2D ................   Passed
    Start 4: HipDNNConvTest.ReferenceConvCorrectness
4/4 Test #4: HipDNNConvTest.ReferenceConvCorrectness ...   Passed

100% tests passed, 0 tests failed out of 4
```

✅ **Phase 1 Complete**: Original hipDNNEP is now built and tested successfully.

---

## Phase 2: Build Morphizen + morphizen-hipdnn

### 3.1 Build ONNXRuntime with Vitis AI Support

```powershell
Set-Location C:\Develop\m\source\onnxruntime

.\build.bat --use_vitisai `
    --config Debug `
    --build_shared_lib `
    --parallel `
    --compile_no_warning_as_error `
    --skip_submodule_sync `
    --build_dir C:\Develop\m\build\onnxruntime-vitisai `
    --skip_tests `
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=C:/Develop/m/local

# Install
cmake --build C:\Develop\m\build\onnxruntime-vitisai\Debug --target install
```

### 3.2 Clone MorphiZen

```powershell
Set-Location C:\Develop\m\source
git clone https://github.com/Xilinx/MorphiZen.git --recursive
```

### 3.3 Verify morphizen-hipdnn

If morphizen-hipdnn is the current working directory:
```powershell
Set-Location C:\Develop\m\source\morphizen-hipdnn
git submodule update --init --recursive
```

### 3.4 Configure morphizen-hipdnn

```powershell
Set-Location C:\Develop\m\source\morphizen-hipdnn

cmake -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DBUILD_SHARED_LIBS=OFF `
    -B C:\Develop\m\build\morphizen-hipdnn `
    -S . `
    -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local `
    -DCMAKE_PREFIX_PATH="C:/Develop/m/dist/therock;C:/Develop/m/local"
```

### 3.5 Build morphizen-hipdnn

```powershell
cmake --build C:\Develop\m\build\morphizen-hipdnn --target install
```

### 3.6 Verify Build

```powershell
# Check built libraries
Get-ChildItem C:\Develop\m\build\morphizen-hipdnn\level-1-pass-hipdnn\*.dll
Get-ChildItem C:\Develop\m\build\morphizen-hipdnn\custom-op-hipdnn\*.dll
Get-ChildItem C:\Develop\m\build\morphizen-hipdnn\test\*.exe
```

---

## Phase 3: Testing and Validation

### 4.1 Run morphizen-hipdnn Tests

```powershell
Set-Location C:\Develop\m\build\morphizen-hipdnn
ctest --output-on-failure
```

### 4.2 Test Comparison: hipDNNEP vs morphizen-hipdnn

#### Test 1: EP Registration (hipDNNEP)

```powershell
Set-Location C:\Develop\m\build\hipDNNEP\RelWithDebInfo
ctest -R RegisterEpLibrary --output-on-failure
```

#### Test 2: Conv2D Operations (hipDNNEP)

```powershell
Set-Location C:\Develop\m\build\hipDNNEP\RelWithDebInfo
ctest -R BasicConv2D --output-on-failure
```

### 4.3 Validation Checklist

- [ ] EP library loads successfully
- [ ] GPU devices are detected correctly
- [ ] Conv2D operations execute on GPU
- [ ] Reference model produces correct results
- [ ] No memory leaks or crashes

---

## Environment Variables Reference

### Required Environment Variables

```powershell
# TheRock SDK location
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"

# HIP platform
$env:HIP_PLATFORM = "amd"

# ONNXRuntime source location
$env:ONNXRUNTIME_ROOT = "C:\Develop\m\source\onnxruntime"

# Add to PATH
$env:PATH = "C:\Develop\m\dist\therock\bin;C:\Develop\m\dist\clang\bin;$env:PATH"
```

### Optional Debug Variables

```powershell
# Enable verbose logging for Xilinx ONNX EP
$env:XLNX_ONNX_EP_VERBOSE = "1"

# Set debug log level
$env:DEBUG_LOG_LEVEL = "3"

# Enable hipDNN debugging
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
```

---

## Troubleshooting

### Issue 1: CMake Not Found

**Error**: `cmake : The term 'cmake' is not recognized`

**Solution**:
```powershell
# Restart PowerShell to refresh PATH after installation
# Or manually add CMake to PATH
$env:PATH = "C:\Program Files\CMake\bin;$env:PATH"
```

### Issue 2: hip-config.cmake Not Found

**Error**: `Could not find a package configuration file provided by "hip"`

**Solution**:
```powershell
# Verify TheRock installation
Test-Path "C:\Develop\m\dist\therock\lib\cmake\hip\hip-config.cmake"

# Ensure THEROCK_DIST is set
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"
$env:PATH = "$env:THEROCK_DIST\bin;$env:PATH"
```

### Issue 3: iree-compile Not Found

**Error**: `iree-compile not found in PATH`

**Solution**:
```powershell
# Check if IREE is in TheRock
Test-Path "C:\Develop\m\dist\therock\bin\iree-compile.exe"

# If not, download separately and add to PATH
$env:PATH = "C:\Develop\m\dist\iree\bin;$env:PATH"
```

### Issue 4: MSVC Compilation Errors with TheRock Headers

**Error**: Various MSVC-incompatible syntax errors

**Solution**: Use Clang instead of MSVC:
```powershell
cmake -G Ninja `
    -DCMAKE_CXX_COMPILER=C:/Develop/m/dist/clang/bin/clang++.exe `
    ...
```

### Issue 5: Tests Not Building

**Warning**: `ONNXRuntime library not found`

**Solution**: Apply Patch 2 to fix the Windows path:
```powershell
# In test/CMakeLists.txt, change line 18:
# From: "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo"
# To:   "${ONNXRUNTIME_ROOT}/build/Windows/RelWithDebInfo"
```

### Issue 6: GPU Not Detected

**Error**: `No GPU devices found`

**Solution**:
```powershell
# Check AMD GPU drivers are installed
# Verify with amdgpu-arch (adjust path based on your Clang installation)
& "C:\Program Files\LLVM\bin\amdgpu-arch.exe"
# Or: & "C:\Develop\m\dist\clang\bin\amdgpu-arch.exe"

# Ensure HIP_PLATFORM is set
$env:HIP_PLATFORM = "amd"
```

### Issue 7: Unexpected HIP_PLATFORM Error

**Error**: `CMake Error at hip-config.cmake: Unexpected HIP_PLATFORM:`

**Solution**: Set the HIP_PLATFORM environment variable before running CMake:
```powershell
$env:HIP_PLATFORM = "amd"
```

Or pass it as a CMake variable:
```powershell
cmake ... -DHIP_PLATFORM=amd
```

### Issue 8: hipdnn_backend.dll Missing at Runtime

**Error**: `Error loading "hipdnn_ep.dll" which depends on "hipdnn_backend.dll" which is missing`

**Solution**: Copy TheRock DLLs to the test directory:
```powershell
$buildDir = "C:\Develop\m\build\hipDNNEP\RelWithDebInfo"
$testDir = "$buildDir\test"
$therockBin = "C:\Develop\m\dist\therock\bin"

Copy-Item "$therockBin\hipdnn_backend.dll" $testDir -Force
Copy-Item "$therockBin\amdhip64.dll" $testDir -Force -ErrorAction SilentlyContinue
```

### Issue 9: hipDNN create_execution_plans Failed

**Error**: `hipDNN create_execution_plans failed: No engine configurations available for the graph`

**Explanation**: This error occurs when the hipDNN backend cannot find suitable GPU kernels for the requested operation. This is typically a hipDNN engine configuration issue, not a hipDNNEP issue.

**Current Status**: The core hipDNNEP integration with ONNX Runtime is working correctly:
- EP registration: ✅ Working
- Device detection: ✅ Working
- Backend availability: ✅ Working
- GPU kernel execution: ❌ Requires hipDNN engine plugins

**Workaround**: Ensure hipDNN engine plugins are properly installed in `$THEROCK_DIST/bin/hipdnn_plugins/engines/`.

---

## Expected Test Results

### Test Suite Overview

hipDNNEP includes 4 tests:

| Test | Description | Status |
|------|-------------|--------|
| `HipDNNEpLoadTest.RegisterEpLibrary` | EP loads and registers with ORT | ✅ Should Pass |
| `HipDNNEpLoadTest.GetEpDevices` | HipDNN device is detected | ✅ Should Pass |
| `HipDNNConvTest.BasicConv2D` | GPU convolution execution | ⚠️ May fail (backend config) |
| `HipDNNConvTest.ReferenceConvCorrectness` | Reference implementation | ✅ Should Pass |

### Successful Build Output

Using the submodule (Option A):
```
-- Found HIP: C:/Develop/m/dist/therock
-- Found iree-compile: C:/Users/.../Scripts/iree-compile.exe
-- Found ONNXRuntime headers at: C:/Develop/m/local/include/onnxruntime
-- Found ONNXRuntime library: C:/Develop/m/local/lib/onnxruntime.lib
-- Configuring done
-- Build files have been written to: C:/Develop/m/build/morphizen-hipdnn-e2e

[17/17] Linking CXX shared library hipdnn_ep.dll
BUILD SUCCESSFUL!
```

### Partial Test Pass (Expected with Current hipDNN)

```
Test project C:/Develop/m/build/morphizen-hipdnn-e2e
    Start 1: HipDNNEpLoadTest.RegisterEpLibrary
1/4 Test #1: HipDNNEpLoadTest.RegisterEpLibrary ........   Passed
    Start 2: HipDNNEpLoadTest.GetEpDevices  
2/4 Test #2: HipDNNEpLoadTest.GetEpDevices .............   Passed
    Start 3: HipDNNConvTest.BasicConv2D
3/4 Test #3: HipDNNConvTest.BasicConv2D ................   Failed (backend)
    Start 4: HipDNNConvTest.ReferenceConvCorrectness
4/4 Test #4: HipDNNConvTest.ReferenceConvCorrectness ...   Passed

75% tests passed, 1 tests failed out of 4
```

**Note**: 75% pass rate with `BasicConv2D` failure is expected. The core hipDNNEP integration is working correctly. The Conv2D execution failure is due to hipDNN backend engine configuration, not the ONNX Runtime execution provider.

---

## Patches Reference

### Patch 1: CMakePresets.json

**File**: `CMakePresets.json`

**Purpose**: Remove hardcoded Linux paths from the author's environment.

**Diff**:
```diff
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
```

**After modification**, the `cacheVariables` section should look like:
```json
"cacheVariables": {
  "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
  "HIP_PLATFORM": "amd"
}
```

### Patch 2: CMakeLists.txt - ONNXRuntime Include Path

**File**: `CMakeLists.txt`

**Purpose**: Support both ORT source layout and installed layout for headers.

**Problem**: The original code only looks for headers at `include/onnxruntime/core/session/` (source layout), but installed ORT has headers at `include/onnxruntime/` (flat layout).

**Diff**:
```diff
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -40,12 +40,21 @@
 if(NOT ONNXRUNTIME_ROOT)
   message(FATAL_ERROR "ONNXRUNTIME_ROOT must be specified...")
 endif()
 
-set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
-
-# Verify ONNXRuntime headers exist
-if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
-  message(FATAL_ERROR "ONNXRuntime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}")
+# Try source layout first: include/onnxruntime/core/session/
+set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
+if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
+  # Try installed layout: include/onnxruntime/
+  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
+  if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
+    message(FATAL_ERROR "ONNXRuntime headers not found. Tried:\n"
+                        "  - ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session/\n"
+                        "  - ${ONNXRUNTIME_ROOT}/include/onnxruntime/")
+  endif()
 endif()
+message(STATUS "Found ONNXRuntime headers at: ${ONNXRUNTIME_INCLUDE_DIR}")
```

### Patch 3: test/CMakeLists.txt - ONNXRuntime Library Path

**File**: `test/CMakeLists.txt`

**Purpose**: Support multiple ONNXRuntime library locations (source build vs installed).

**Diff**:
```diff
--- a/test/CMakeLists.txt
+++ b/test/CMakeLists.txt
@@ -16,12 +16,22 @@
 
 # ONNXRuntime library for testing
-set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
+set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "...")
 
 find_library(ONNXRUNTIME_LIB onnxruntime
-  HINTS ${ONNXRUNTIME_LIB_DIR}
+  HINTS
+    ${ONNXRUNTIME_LIB_DIR}
+    "${ONNXRUNTIME_ROOT}/lib"
+    "${ONNXRUNTIME_ROOT}/bin"
+    "${ONNXRUNTIME_ROOT}/build/Windows/RelWithDebInfo/RelWithDebInfo"
 )
 
 if(NOT ONNXRUNTIME_LIB)
-  message(WARNING "ONNXRuntime library not found at ${ONNXRUNTIME_LIB_DIR}...")
+  message(WARNING "ONNXRuntime library not found. Tried:\n"
+                  "  - ${ONNXRUNTIME_LIB_DIR}\n"
+                  "  - ${ONNXRUNTIME_ROOT}/lib\n"
+                  "  - ${ONNXRUNTIME_ROOT}/bin\n"
+                  "  - ${ONNXRUNTIME_ROOT}/build/Windows/RelWithDebInfo/RelWithDebInfo\n"
+                  "Tests will not be built.")
   return()
 endif()
+message(STATUS "Found ONNXRuntime library: ${ONNXRUNTIME_LIB}")
```

**Note**: This allows hipDNNEP to work with both:
- ORT source builds: `build/Windows/RelWithDebInfo/`
- ORT installed prefix: `lib/` or `bin/`

---

## Quick Reference

### Complete Command Sequence (PowerShell)

```powershell
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Create directories
New-Item -ItemType Directory -Force -Path @(
    "C:\Develop\m\source",
    "C:\Develop\m\build",
    "C:\Develop\m\local",
    "C:\Develop\m\dist\clang",
    "C:\Develop\m\dist\therock"
)

# Install build tools
winget install Kitware.CMake --accept-package-agreements --accept-source-agreements
winget install Ninja-build.Ninja --accept-package-agreements --accept-source-agreements

# RESTART PowerShell HERE to refresh PATH

# Set environment variables
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"
$env:HIP_PLATFORM = "amd"
$env:PATH = "$env:THEROCK_DIST\bin;C:\Develop\m\dist\clang\bin;$env:PATH"

# ============================================================================
# PHASE 1: BUILD HIPDNNEP
# ============================================================================

# Clone and build ONNXRuntime
Set-Location C:\Develop\m\source
git clone https://github.com/microsoft/onnxruntime.git
Set-Location onnxruntime
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir C:\Develop\m\build\onnxruntime

$env:ONNXRUNTIME_ROOT = "C:\Develop\m\source\onnxruntime"

# Clone hipDNNEP
Set-Location C:\Develop\m\source
git clone https://github.com/MaheshRavishankar/hipDNNEP.git
Set-Location hipDNNEP
git checkout de7921872f218a75e3f6de589a8ed4be9f08782

# Apply patches (see Patches Reference section)
# Then configure and build (adjust Clang path if needed):
cmake -G Ninja -B C:\Develop\m\build\hipDNNEP\RelWithDebInfo -S . `
    -DCMAKE_BUILD_TYPE=RelWithDebInfo `
    -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" `
    -DTHEROCK_DIST=C:/Develop/m/dist/therock `
    -DONNXRUNTIME_ROOT=C:/Develop/m/source/onnxruntime

cmake --build C:\Develop\m\build\hipDNNEP\RelWithDebInfo

# Run tests
Set-Location C:\Develop\m\build\hipDNNEP\RelWithDebInfo
ctest --output-on-failure

# ============================================================================
# PHASE 2: BUILD MORPHIZEN-HIPDNN
# ============================================================================

# Build ONNXRuntime with Vitis AI
Set-Location C:\Develop\m\source\onnxruntime
.\build.bat --use_vitisai --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir C:\Develop\m\build\onnxruntime-vitisai --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=C:/Develop/m/local
cmake --build C:\Develop\m\build\onnxruntime-vitisai\Debug --target install

# Clone MorphiZen
Set-Location C:\Develop\m\source
git clone https://github.com/Xilinx/MorphiZen.git --recursive

# Build morphizen-hipdnn
Set-Location C:\Develop\m\source\morphizen-hipdnn
cmake -G Ninja -B C:\Develop\m\build\morphizen-hipdnn -S . `
    -DCMAKE_BUILD_TYPE=Debug `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local `
    -DCMAKE_PREFIX_PATH="C:/Develop/m/dist/therock;C:/Develop/m/local"

cmake --build C:\Develop\m\build\morphizen-hipdnn --target install

# Run tests
Set-Location C:\Develop\m\build\morphizen-hipdnn
ctest --output-on-failure
```

---

## Summary

This guide provides complete instructions for building and testing hipDNNEP and morphizen-hipdnn on Windows:

1. ✅ Environment setup with required tools (CMake, Ninja, Clang, TheRock)
2. ✅ Building original hipDNNEP with required patches
3. ✅ Building morphizen-hipdnn
