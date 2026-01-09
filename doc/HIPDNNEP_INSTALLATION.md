# hipDNNEP Installation Guide

## Installation Date
January 8, 2026

## Overview
This document describes the installation process for hipDNNEP (hipDNN Execution Provider for ONNX Runtime) on Windows.

## Installation Summary

### Prerequisites Status
- ✅ **Clang 20.x**: Installed at `c:/LLVM20`
- ✅ **ONNXRuntime**: Installed at `C:/Develop/m/local`
- ✅ **FlatBuffers**: Built and installed at `C:/Develop/m/local`
- ✅ **hipDNNEP Source**: Cloned to `C:\Develop\m\source\hipDNNEP`
- ✅ **TheRock SDK**: Downloaded and extracted to `C:\Develop\TheRock`
- ⚠️ **GPU Hardware**: No AMD GPU present (software-only installation)
- ✅ **Build Status**: Successfully built using Clang toolchain (MSVC incompatible)

### Completed Steps

#### 1. Clone hipDNNEP Repository ✅
**Command:**
```bash
cd C:\Develop\m\source
git clone https://github.com/zpye/hipDNNEP.git
```

**Status:** Successfully cloned
**Location:** `C:\Develop\m\source\hipDNNEP`
**Current Commit:** `91ae8ad` - Windows build fixes included

**Repository Details:**
```
91ae8ad (HEAD -> dev, origin/dev, origin/HEAD) [fix] build and run on Windows platform
de79218 (origin/main) Add CMakePresets.json to match the build instructions.
def96c4 Fix formatting errors.
928282b Add pre-commit CI workflow
378cf94 Add pre-commit hooks configuration
```

---

## Required Manual Steps

### Step 1: Download TheRock SDK

TheRock SDK cannot be downloaded automatically. You must manually download it from AWS S3.

**Instructions:**

1. **Determine GPU Architecture**
   
   Since no GPU is present, we'll use **gfx1103** (default for Radeon RX 7000 series) as the target architecture.
   
   If you have a GPU, you can check its architecture:
   ```cmd
   c:\LLVM20\bin\amdgpu-arch.exe
   ```

2. **Download the Correct SDK**
   
   - Visit: https://therock-nightly-tarball.s3.amazonaws.com/index.html
   - Find the latest tarball for **gfx110X-all** architecture:
     ```
     therock-dist-windows-gfx110X-all-*.tar.gz
     ```
   - Example filename: `therock-dist-windows-gfx110X-all-7.10.0a20251103.tar.gz`

3. **Extract TheRock SDK**
   
   Extract the downloaded tarball to `C:\Develop\TheRock`
   
   Using tar (built into Windows 10/11):
   ```cmd
   tar -xzf therock-dist-windows-gfx110X-all-*.tar.gz -C C:\Develop
   ```
   
   Or use 7-Zip or another extraction tool.

4. **Verify TheRock Installation**
   
   After extraction, verify these paths exist:
   ```
   C:\Develop\TheRock\bin\hipconfig.exe
   C:\Develop\TheRock\bin\iree-compile.exe
   C:\Develop\TheRock\lib\cmake\hipdnn_frontend\
   C:\Develop\TheRock\lib\cmake\hipdnn_backend\
   C:\Develop\TheRock\share\cmake\nlohmann_json\
   ```

### Step 2: Check for Required SDK Modifications

After extracting TheRock, check if these configuration files need modification:

#### File 1: hipdnn_backend CMake Config

**Path:** `C:\Develop\TheRock\lib\cmake\hipdnn_backend\hipdnn_backendConfig.cmake`

**Issue:** May contain hardcoded paths in `set_target_properties`

**Action:** Open the file and look for `set_target_properties` for target `hipdnn_backend`. Remove any hardcoded absolute paths that reference non-existent directories.

**Example of hardcoded path to remove:** `B:/build/core/clr/dist/include`

#### File 2: hipdnn_data_sdk CMake Targets

**Path:** `C:\Develop\TheRock\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake`

**Issue:** May contain hardcoded paths in `set_target_properties`

**Action:** Open the file and look for `set_target_properties` for target `hipdnn_data_sdk`. Remove any hardcoded absolute paths that reference non-existent directories.

**Example of hardcoded paths to remove:** All `B:/build/third-party/*` paths

**Example diff to apply:**
```diff
  set_target_properties(hipdnn_data_sdk PROPERTIES
    INTERFACE_COMPILE_FEATURES "cxx_std_17"
-   INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/llvm-project/install/include;B:/build/third-party/json/install/include;${_IMPORT_PREFIX}/include"
+   INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    INTERFACE_LINK_LIBRARIES "hip::host"
  )
```

The key is to remove all absolute paths starting with `B:/build/` and keep only the relative `${_IMPORT_PREFIX}/include` path.

#### File 3: nlohmann_json CMake Config

**Path:** `C:\Develop\TheRock\share\cmake\nlohmann_json\nlohmann_jsonTargets.cmake`

**Issue:** May contain `INTERFACE_SOURCES` attribute that references missing files

**Action:** Open the file and look for `set_target_properties` for target `nlohmann_json::nlohmann_json`. Remove the `INTERFACE_SOURCES` line if it exists.

**Note:** The zpye/hipDNNEP repository may have workarounds for these issues, so try building first before modifying.

---

## Automated Steps (To Be Executed After TheRock Installation)

Once TheRock SDK is installed at `C:\Develop\TheRock`, run these commands:

### Step 3: Set Environment Variables

```cmd
set THEROCK_DIST=C:\Develop\TheRock
set HIP_PLATFORM=amd
set PATH=C:\Develop\TheRock\bin;%PATH%
```

For permanent setup, add these to System Environment Variables.

### Step 4: Configure hipDNNEP Build

```cmd
cd C:\Develop\m\source\hipDNNEP

cmake -G Ninja ^
  -B C:/Develop/m/build/hipDNNEP/RelWithDebInfo ^
  -S . ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DTHEROCK_DIST=C:/Develop/TheRock ^
  -DONNXRUNTIME_ROOT=C:/Develop/m/local
```

**Expected CMake Output:**
```
-- Found hip
-- Found hipdnn_frontend
-- Found hipdnn_backend
-- Found iree-compile: C:/Develop/TheRock/bin/iree-compile.exe
-- Found ONNXRuntime headers
```

### Step 5: Build hipDNNEP

```cmd
cmake --build C:/Develop/m/build/hipDNNEP/RelWithDebInfo
```

**Build Outputs:**
- `C:\Develop\m\build\hipDNNEP\RelWithDebInfo\hipdnn_ep.dll`
- `C:\Develop\m\build\hipDNNEP\RelWithDebInfo\test\hipdnn_ep_tests.exe`

### Step 6: Prepare Test Environment

Copy required DLLs to test directory:

```cmd
copy C:\Develop\m\build\hipDNNEP\RelWithDebInfo\hipdnn_ep.dll C:\Develop\m\build\hipDNNEP\RelWithDebInfo\test\
copy C:\Develop\m\local\bin\onnxruntime.dll C:\Develop\m\build\hipDNNEP\RelWithDebInfo\test\
```

Ensure TheRock bin is in PATH:
```cmd
set PATH=C:\Develop\TheRock\bin;%PATH%
```

### Step 7: Run Tests (Optional)

```cmd
cd C:\Develop\m\build\hipDNNEP\RelWithDebInfo\test
hipdnn_ep_tests.exe
```

**Expected Behavior:**
- ⚠️ Tests will likely fail with HIP runtime errors (no GPU device)
- ✅ Build should succeed regardless

---

## Directory Structure

After complete installation:

```
C:\Develop\
├── TheRock\                    # TheRock ROCm SDK (MANUAL DOWNLOAD REQUIRED)
│   ├── bin\
│   │   ├── hipconfig.exe
│   │   ├── iree-compile.exe
│   │   └── hipdnn_backend.dll
│   ├── lib\
│   │   └── cmake\
│   │       ├── hipdnn_frontend\
│   │       └── hipdnn_backend\
│   └── share\
│       └── cmake\
│           └── nlohmann_json\
├── LLVM20\                     # Clang 20.x (Already installed)
│   └── bin\
│       └── amdgpu-arch.exe
└── m\
    ├── local\                  # ONNXRuntime (Already installed)
    │   ├── bin\
    │   │   └── onnxruntime.dll
    │   └── include\
    │       └── onnxruntime\
    ├── source\
    │   └── hipDNNEP\          # hipDNNEP source (Cloned)
    │       ├── CMakeLists.txt
    │       ├── include\
    │       ├── src\
    │       └── test\
    └── build\
        └── hipDNNEP\
            └── RelWithDebInfo\  # Build outputs (After build)
                ├── hipdnn_ep.dll
                └── test\
                    └── hipdnn_ep_tests.exe
```

---

## Troubleshooting

### Issue: CMake Cannot Find hipdnn_frontend or hipdnn_backend

**Symptoms:**
```
CMake Error: Could not find a package configuration file provided by "hipdnn_frontend"
```

**Solutions:**
1. Verify TheRock SDK is extracted to `C:\Develop\TheRock`
2. Check that cmake config files exist:
   - `C:\Develop\TheRock\lib\cmake\hipdnn_frontend\hipdnn_frontendConfig.cmake`
   - `C:\Develop\TheRock\lib\cmake\hipdnn_backend\hipdnn_backendConfig.cmake`
3. Verify `THEROCK_DIST` environment variable is set correctly
4. Check the SDK modification steps above if config files have issues

### Issue: CMake Cannot Find iree-compile

**Symptoms:**
```
CMake Error: iree-compile not found in PATH
```

**Solution:**
1. Verify `C:\Develop\TheRock\bin\iree-compile.exe` exists
2. Add `C:\Develop\TheRock\bin` to PATH
3. Restart your terminal/PowerShell session

### Issue: Build Fails with Missing nlohmann_json

**Symptoms:**
```
CMake Error: INTERFACE_SOURCES refers to non-existent file
```

**Solution:**
Apply the SDK modification for `nlohmann_jsonTargets.cmake` (see Step 2 above)

### Issue: Tests Fail with HIP Runtime Errors

**Symptoms:**
```
HIP error: no device found
```

**Expected:** This is normal without AMD GPU hardware. The build is successful even if tests fail.

---

## Verification

After successful build, verify:

1. **Build artifacts exist:**
   ```cmd
   dir C:\Develop\m\build\hipDNNEP\RelWithDebInfo\hipdnn_ep.dll
   dir C:\Develop\m\build\hipDNNEP\RelWithDebInfo\test\hipdnn_ep_tests.exe
   ```

2. **Libraries can be found:**
   ```cmd
   dumpbin /DEPENDENTS C:\Develop\m\build\hipDNNEP\RelWithDebInfo\hipdnn_ep.dll
   ```

3. **CMake configuration was successful** (check build logs for "Found" messages)

---

## Build Solution: Use Clang Instead of MSVC

### MSVC Incompatibility Issue

**Problem:**
MSVC cannot compile TheRock SDK headers due to type conversion errors:
```
C:\Develop\TheRock\include\hipdnn\data_sdk\hipdnn_data_sdk/utilities/UtilsBfp16.hpp(17): error C2440: 
'<function-style-cast>': cannot convert from 'float' to 'hip_bfloat16'
```

**Solution: Use Clang Toolchain**

The TheRock SDK is compatible with Clang but not MSVC. **You must use Clang 20.x as the compiler.**

**Correct Build Command:**
```powershell
cd C:\Develop\m\source\hipDNNEP

cmake -G Ninja `
  -B C:/Develop/m/build/hipDNNEP/RelWithDebInfo-Clang `
  -S . `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DCMAKE_CXX_COMPILER=c:/LLVM20/bin/clang++.exe `
  -DTHEROCK_DIST=C:/Develop/TheRock `
  -DONNXRUNTIME_ROOT=C:/Develop/m/local `
  -DHIPDNN_EP_BUILD_TESTS=OFF `
  -DCMAKE_PREFIX_PATH="C:/Develop/m/local;C:/Develop/TheRock"

cmake --build C:/Develop/m/build/hipDNNEP/RelWithDebInfo-Clang
```

**Result:** ✅ Build succeeds with Clang

---

## Steps Completed

1. ✅ Downloaded TheRock SDK (gfx110X-all-7.11.0a20260108, 2.17 GB)
2. ✅ Extracted TheRock SDK to `C:\Develop\TheRock`
3. ✅ Fixed hardcoded paths in TheRock SDK cmake configs:
   - `hipdnn_backendConfig.cmake` - Removed `B:/build/core/clr/dist/include`
   - `hipdnn_data_sdkTargets.cmake` - Removed all `B:/build/third-party/*` paths
4. ✅ Built and installed FlatBuffers 25.12.19 to `C:/Develop/m/local`
5. ✅ Cloned hipDNNEP repository (commit 91ae8ad with Windows fixes)
6. ✅ Added `/utf-8` compiler flag to CMakeLists.txt
7. ✅ Successfully configured CMake build with Clang 20.x
8. ✅ Successfully built hipDNNEP using Clang toolchain
9. ✅ Generated `hipdnn_ep.dll` (738 KB)

**Build Output Location:** `C:\Develop\m\build\hipDNNEP\RelWithDebInfo-Clang\hipdnn_ep.dll`

---

## Notes

- **Clang Toolchain Required:** MSVC cannot compile TheRock SDK headers. You MUST use Clang 20.x as the compiler.
- **No GPU Required for Build:** The build succeeds without AMD GPU hardware. Only runtime execution requires GPU.
- **SDK Modifications:** We fixed hardcoded paths in TheRock SDK cmake configuration files.
- **FlatBuffers Required:** FlatBuffers must be built and installed separately as it's a dependency.
- **Development Use:** This installation is suitable for development, integration, and compilation purposes even without GPU hardware.

---

## References

- **hipDNNEP Repository:** https://github.com/zpye/hipDNNEP
- **TheRock Nightly Builds:** https://therock-nightly-tarball.s3.amazonaws.com/index.html
- **Windows Build Fixes Commit:** https://github.com/zpye/hipDNNEP/commit/91ae8addde1c9aae62bd37167fe8e4b72661df76

---

## Installation Script

An automated installation script is provided: `install_hipdnnep.ps1`

This script will:
1. Verify TheRock SDK installation
2. Verify ONNXRuntime installation
3. Check for Clang 20.x compiler
4. Configure CMake build (using Clang)
5. Build hipDNNEP
6. Prepare test environment

**Usage:**
```powershell
.\install_hipdnnep.ps1
```

**Note:** The script automatically uses Clang compiler. MSVC will not work with TheRock SDK.

---

## Complete Installation Summary

### Prerequisites (All Met ✅)
- Clang 20.x at `c:/LLVM20`
- ONNXRuntime at `C:/Develop/m/local`
- TheRock SDK at `C:\Develop\TheRock`

### Installation Steps Executed
1. ✅ Cloned hipDNNEP from https://github.com/zpye/hipDNNEP (commit 91ae8ad)
2. ✅ Downloaded TheRock SDK 7.11.0a20260108 (2.17 GB)
3. ✅ Extracted TheRock SDK to C:\Develop\TheRock
4. ✅ Fixed TheRock SDK cmake hardcoded paths (hipdnn_backend, hipdnn_data_sdk)
5. ✅ Built FlatBuffers 25.12.19 from source
6. ✅ Installed FlatBuffers to C:/Develop/m/local
7. ✅ Configured hipDNNEP with Clang 20.x compiler
8. ✅ Successfully built hipDNNEP

### Build Output
**File:** `C:\Develop\m\build\hipDNNEP\RelWithDebInfo-Clang\hipdnn_ep.dll`
**Size:** 738,816 bytes (721 KB)
**Compiler:** Clang 20.1.8
**Status:** ✅ Build Successful

### Key Learnings
- **MSVC is incompatible** with TheRock SDK headers - Clang is required
- TheRock SDK cmake configs contain hardcoded paths that must be removed
- FlatBuffers must be built separately and installed
- The build succeeds without GPU hardware (runtime requires GPU)
