# onnx-hipdnn-ep Windows Build Guide

This document provides complete step-by-step instructions for building and testing onnx-hipdnn-ep on Windows with AMD ROCm GPU.

> **Note**: This is the Windows adaptation of the Linux build guide. Key differences from Linux:
> - No SSH required (local execution)
> - PowerShell instead of bash
> - winget instead of apt
> - Windows paths and environment variables

## Table of Contents

1. [Checking AMD GPU Hardware Information](#checking-amd-gpu-hardware-information)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Phase 1: Build ONNXRuntime](#phase-1-build-onnxruntime)
5. [Phase 2: Build Morphizen + onnx-hipdnn-ep](#phase-2-build-morphizen--onnx-hipdnn-ep)
6. [Phase 3: Testing and Validation](#phase-3-testing-and-validation)
7. [Environment Variables Reference](#environment-variables-reference)
8. [Troubleshooting](#troubleshooting)
9. [Quick Reference](#quick-reference)

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

### Method 4: Check TheRock Distribution Version

TheRock includes a version file that shows the distribution version:

```powershell
# Check TheRock distribution version
Get-Content "C:\Develop\m\dist\therock\.info\version"
# Output: 7.11.0
```

For more detailed version information, check the ROCm version header:

```powershell
# Get ROCm build info (includes version and git hash)
Select-String -Path "C:\Develop\m\dist\therock\include\rocm-core\rocm_version.h" -Pattern "ROCM_BUILD_INFO"
# Output: #define ROCM_BUILD_INFO    "7.11.0.2-9999-56870acb4f"
```

#### Understanding Version Numbers

TheRock uses **two different version schemes**:

| Version | What It Is | Example |
|---------|-----------|---------|
| **TheRock Distribution Version** | Package version from `.info/version` | `7.11.0` |
| **HIP Runtime Version** | HIP API version from `hipconfig --version` | `7.2.53150-56870acb4f` |
| **Tarball Version** | Build version in filename | `7.10.0a20251103` |

- **TheRock version** (e.g., `7.11.0`) = The distribution package version
- **HIP version** (e.g., `7.2.53150`) = The HIP runtime component version inside TheRock
- **Tarball suffix** (e.g., `7.10.0a20251103`) = Build identifier with date (the `a` indicates alpha/nightly)

The git hash `56870acb4f` appears in both `ROCM_BUILD_INFO` and HIP version, confirming they're from the same source build.

### GPU Information Summary

| Property | Example 1 (W7900) | Example 2 (8050S) | How to Check |
|----------|-------------------|-------------------|--------------|
| GPU Model | AMD Radeon PRO W7900 | AMD Radeon(TM) 8050S Graphics | PowerShell `Get-CimInstance` |
| GPU Architecture | gfx1100 | gfx1151 | `amdgpu-arch.exe` |
| GFX Family | gfx110X-all | gfx115X-all | See [GFX Family table](#step-2-determine-gfx-family) |
| Device ID | 0x7448 | 0x1586 | PowerShell `Get-CimInstance` |
| VRAM (Windows) | ~4 GB | 512 MB (dynamic) | PowerShell `Get-CimInstance` |
| VRAM (HIP) | N/A | 24.26 GB | `hipInfo.exe` |
| Driver Version | 32.0.21037.1004 | 32.0.22029.9039 | PowerShell `Get-CimInstance` |
| TheRock Version | 7.11.0 | 7.11.0 | `Get-Content "$env:THEROCK_DIST\.info\version"` |
| HIP Version | 7.2.53150 | 7.2.53150-e5316dcbd9 | `hipconfig.exe --version` |
| HIP Platform | amd | amd | `hipconfig.exe --platform` |

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
| Python | 3.x | Build scripts |

### Directory Structure

This guide uses the following directory structure:

```
C:\Develop\m\
├── source\           # Source code repositories
│   ├── MorphiZen\
│   ├── onnx-hipdnn-ep\
│   │   └── 3rd-party\morphizen\  # MorphiZen framework (git submodule)
│   └── onnxruntime\
├── build\            # Build directories
│   ├── onnxruntime\
│   └── onnx-hipdnn-ep\
├── local\            # Installation prefix
│   ├── bin\
│   ├── lib\
│   └── include\
└── dist\             # External tools (read-only/downloaded)
    ├── clang\        # Clang 20.x
    └── therock\      # TheRock SDK
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
    "C:\Develop\m\dist\therock",
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

TheRock provides HIP, hipDNN, and other ROCm components.

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

#### Step 6: Fix TheRock SDK Hardcoded Paths (IMPORTANT)

TheRock SDK CMake config files may contain **hardcoded absolute paths** from the original build machine. These paths typically start with `B:/build/` and must be removed before using the SDK.

##### Detect Hardcoded Paths

Run this PowerShell command to find all files with hardcoded build paths:

```powershell
# Find all cmake files with hardcoded B:/build paths
Get-ChildItem -Path "$env:THEROCK_DIST\lib\cmake" -Recurse -Filter "*.cmake" | 
    Select-String -Pattern "B:/build" | 
    Select-Object Path, LineNumber, Line
```

**Sample output showing affected files:**
```
Path                                                              LineNumber Line
----                                                              ---------- ----
C:\Develop\m\dist\therock\lib\cmake\hipdnn_frontend\..Targets.cmake       12   INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/flatbuffers/dist/include;...
C:\Develop\m\dist\therock\lib\cmake\hipdnn_backend\..Config.cmake         28   INTERFACE_INCLUDE_DIRECTORIES "B:/build/core/clr/dist/include;...
C:\Develop\m\dist\therock\lib\cmake\hipdnn_data_sdk\..Targets.cmake       15   INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/llvm-project/install/include;...
```

##### Files That Commonly Need Fixing

| File | Hardcoded Path to Remove |
|------|--------------------------|
| `hipdnn_frontend/hipdnn_frontendTargets.cmake` | `B:/build/third-party/flatbuffers/dist/include` |
| `hipdnn_backend/hipdnn_backendConfig.cmake` | `B:/build/core/clr/dist/include` |
| `hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake` | `B:/build/third-party/llvm-project/install/include;B:/build/third-party/json/install/include` |

##### Example Fix: hipdnn_frontendTargets.cmake

**File location**: `$THEROCK_DIST\lib\cmake\hipdnn_frontend\hipdnn_frontendTargets.cmake`

**Before** (with hardcoded flatbuffers path):
```cmake
set_target_properties(hipdnn_frontend PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/flatbuffers/dist/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host;hipdnn_data_sdk"
)
```

**After** (fixed - remove the B:/build path):
```cmake
set_target_properties(hipdnn_frontend PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host;hipdnn_data_sdk"
)
```

##### Example Fix: hipdnn_data_sdkTargets.cmake

**File location**: `$THEROCK_DIST\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake`

**Before** (with multiple hardcoded paths):
```cmake
set_target_properties(hipdnn_data_sdk PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/llvm-project/install/include;B:/build/third-party/json/install/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
)
```

**After** (fixed - remove all B:/build paths):
```cmake
set_target_properties(hipdnn_data_sdk PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
)
```

##### Example Fix: hipdnn_backendConfig.cmake

**File location**: `$THEROCK_DIST\lib\cmake\hipdnn_backend\hipdnn_backendConfig.cmake`

Look for any `INTERFACE_INCLUDE_DIRECTORIES` with `B:/build/core/clr/dist/include` and remove it, keeping only `${_IMPORT_PREFIX}/include`.

##### Automated Fix Script

Save this as `fix_therock_paths.ps1` and run it to automatically fix the hardcoded paths:

```powershell
# Fix TheRock SDK Hardcoded Paths
# Usage: .\fix_therock_paths.ps1 -TherockDist "C:\Develop\m\dist\therock"

param(
    [Parameter(Mandatory=$true)]
    [string]$TherockDist
)

$cmakeDir = Join-Path $TherockDist "lib\cmake"

if (-not (Test-Path $cmakeDir)) {
    Write-Error "CMake directory not found: $cmakeDir"
    exit 1
}

Write-Host "Scanning for hardcoded paths in: $cmakeDir" -ForegroundColor Yellow

# Find all cmake files with B:/build paths
$files = Get-ChildItem -Path $cmakeDir -Recurse -Filter "*.cmake" | 
    Where-Object { (Get-Content $_.FullName -Raw) -match "B:/build" }

if ($files.Count -eq 0) {
    Write-Host "No hardcoded paths found. TheRock SDK is clean." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($files.Count) file(s) with hardcoded paths:" -ForegroundColor Yellow
$files | ForEach-Object { Write-Host "  - $($_.FullName)" }

foreach ($file in $files) {
    Write-Host "`nProcessing: $($file.Name)" -ForegroundColor Cyan
    
    # Read file content
    $content = Get-Content $file.FullName -Raw
    
    # Pattern: Remove B:/build paths from INTERFACE_INCLUDE_DIRECTORIES
    # Matches: "B:/build/...;${_IMPORT_PREFIX}..." -> "${_IMPORT_PREFIX}..."
    $pattern = '"B:/build[^"]*;(\$\{_IMPORT_PREFIX\}/include)"'
    $replacement = '"$1"'
    
    $newContent = $content -replace $pattern, $replacement
    
    # Also handle case where B:/build path is the only remaining path after semicolon
    $pattern2 = ';B:/build[^";]*'
    $newContent = $newContent -replace $pattern2, ''
    
    # Handle case where B:/build is at the start followed by semicolon
    $pattern3 = '"B:/build[^";]*;'
    $newContent = $newContent -replace $pattern3, '"'
    
    if ($content -ne $newContent) {
        # Create backup
        $backupPath = "$($file.FullName).bak"
        Copy-Item $file.FullName $backupPath -Force
        Write-Host "  Created backup: $backupPath" -ForegroundColor Gray
        
        # Write fixed content
        Set-Content $file.FullName $newContent -NoNewline
        Write-Host "  Fixed: $($file.Name)" -ForegroundColor Green
    } else {
        Write-Host "  No changes needed" -ForegroundColor Gray
    }
}

Write-Host "`nDone! Verify the fixes by running CMake configure again." -ForegroundColor Green
```

**Usage:**
```powershell
.\fix_therock_paths.ps1 -TherockDist "C:\Develop\m\dist\therock"
```

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

---

## Phase 1: Build ONNXRuntime

### Build ONNXRuntime

### Build ONNXRuntime with Vitis AI Support

```powershell
Set-Location C:\Develop\m\source\onnxruntime

.\build.bat `
    --config Debug `
    --build_shared_lib `
    --parallel `
    --compile_no_warning_as_error `
    --skip_submodule_sync `
    --build_dir C:\Develop\m\build\onnxruntime `
    --skip_tests `
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=C:/Develop/m/local

# Install
cmake --build C:\Develop\m\build\onnxruntime\Debug --target install
```

---

## Phase 2: Build Morphizen + onnx-hipdnn-ep

### 3.1 Clone MorphiZen

```powershell
Set-Location C:\Develop\m\source
git clone https://github.com/Xilinx/MorphiZen.git --recursive
```

### 3.2 Clone onnx-hipdnn-ep

```powershell
Set-Location C:\Develop\m\source
git clone --recursive https://github.com/ROCm/onnx-hipdnn-ep.git
```

> **Note**: The `--recursive` flag is required to clone the `3rd-party/morphizen` submodule. If you already cloned without it, run:
> ```powershell
> Set-Location C:\Develop\m\source\onnx-hipdnn-ep
> git submodule update --init --recursive
> ```

### 3.3 Configure onnx-hipdnn-ep

```powershell
Set-Location C:\Develop\m\source\onnx-hipdnn-ep

cmake -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DBUILD_SHARED_LIBS=OFF `
    -B C:\Develop\m\build\onnx-hipdnn-ep `
    -S . `
    -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local `
    -DCMAKE_PREFIX_PATH="C:/Develop/m/dist/therock;C:/Develop/m/local"
```

### 3.4 Build onnx-hipdnn-ep

```powershell
cmake --build C:\Develop\m\build\onnx-hipdnn-ep --target install
```

### 3.5 Verify Build

```powershell
# Check built libraries
Get-ChildItem C:\Develop\m\build\onnx-hipdnn-ep\level-1-pass-hipdnn\*.dll
Get-ChildItem C:\Develop\m\build\onnx-hipdnn-ep\custom-op-hipdnn\*.dll
Get-ChildItem C:\Develop\m\build\onnx-hipdnn-ep\test\*.exe
```

---

## Phase 3: Testing and Validation

### 4.1 Run onnx-hipdnn-ep Tests

```powershell
Set-Location C:\Develop\m\build\onnx-hipdnn-ep
ctest --output-on-failure
```

### 4.2 Validation Checklist

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

### Issue 3: MSVC Compilation Errors with TheRock Headers

**Error**: Various MSVC-incompatible syntax errors

**Solution**: Use Clang instead of MSVC:
```powershell
cmake -G Ninja `
    -DCMAKE_CXX_COMPILER=C:/Develop/m/dist/clang/bin/clang++.exe `
    ...
```

### Issue 4: Tests Not Building

**Warning**: `ONNXRuntime library not found`

**Solution**: Apply Patch 2 to fix the Windows path:
```powershell
# In test/CMakeLists.txt, change line 18:
# From: "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo"
# To:   "${ONNXRUNTIME_ROOT}/build/Windows/RelWithDebInfo"
```

### Issue 5: GPU Not Detected

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

### Issue 6: Unexpected HIP_PLATFORM Error

**Error**: `CMake Error at hip-config.cmake: Unexpected HIP_PLATFORM:`

**Solution**: Set the HIP_PLATFORM environment variable before running CMake:
```powershell
$env:HIP_PLATFORM = "amd"
```

Or pass it as a CMake variable:
```powershell
cmake ... -DHIP_PLATFORM=amd
```

### Issue 7: TheRock SDK Contains Hardcoded Build Paths

**Error**:
```
CMake Error in level-1-pass-hipdnn/CMakeLists.txt:
  Imported target "hipdnn_frontend" includes non-existent path
    "B:/build/third-party/flatbuffers/dist/include"
```

**Cause**: TheRock SDK CMake configuration files contain hardcoded absolute paths from the original build machine (typically starting with `B:/build/`). These paths don't exist on your system.

**Affected Files** (may vary by TheRock version):
- `lib/cmake/hipdnn_frontend/hipdnn_frontendTargets.cmake`
- `lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake`
- `lib/cmake/hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake`

**Solution**: See [Step 6: Fix TheRock SDK Hardcoded Paths](#step-6-fix-therock-sdk-hardcoded-paths-important) in the Environment Setup section for detailed instructions and an automated fix script.

**Quick Fix** (for the specific flatbuffers error):

1. Open the file:
   ```powershell
   code "$env:THEROCK_DIST\lib\cmake\hipdnn_frontend\hipdnn_frontendTargets.cmake"
   ```

2. Find the line with `INTERFACE_INCLUDE_DIRECTORIES` that contains `B:/build/third-party/flatbuffers/dist/include`

3. Remove the hardcoded path, keeping only `${_IMPORT_PREFIX}/include`:
   ```cmake
   # Before:
   INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/flatbuffers/dist/include;${_IMPORT_PREFIX}/include"
   
   # After:
   INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
   ```

4. Save and re-run CMake configure.

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
# PHASE 1: BUILD ONNXRuntime
# ============================================================================

# Clone and build ONNXRuntime
Set-Location C:\Develop\m\source
git clone https://github.com/microsoft/onnxruntime.git
Set-Location onnxruntime
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir C:\Develop\m\build\onnxruntime


# ============================================================================
# PHASE 2: BUILD onnx-hipdnn-ep
# ============================================================================

# Build ONNXRuntime with Vitis AI
Set-Location C:\Develop\m\source\onnxruntime
.\build.bat --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir C:\Develop\m\build\onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=C:/Develop/m/local
cmake --build C:\Develop\m\build\onnxruntime\Debug --target install

# Clone MorphiZen
Set-Location C:\Develop\m\source
git clone https://github.com/Xilinx/MorphiZen.git --recursive

# Clone onnx-hipdnn-ep (with submodule)
git clone --recursive https://github.com/ROCm/onnx-hipdnn-ep.git

# Build onnx-hipdnn-ep
Set-Location C:\Develop\m\source\onnx-hipdnn-ep
cmake -G Ninja -B C:\Develop\m\build\onnx-hipdnn-ep -S . `
    -DCMAKE_BUILD_TYPE=Debug `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local `
    -DCMAKE_PREFIX_PATH="C:/Develop/m/dist/therock;C:/Develop/m/local"

cmake --build C:\Develop\m\build\onnx-hipdnn-ep --target install

# Run tests
Set-Location C:\Develop\m\build\onnx-hipdnn-ep
ctest --output-on-failure
```

---

## Summary

This guide provides complete instructions for building and testing  onnx-hipdnn-ep on Windows:

1. ✅ Environment setup with required tools (CMake, Ninja, Clang, TheRock)
2. ✅ Building ONNXRuntime with required patches
3. ✅ Building onnx-hipdnn-ep
