# morphizen-hipdnn Build and Installation Work Log

**Date:** January 12, 2026  
**Operator:** mingyue  
**Objective:** Build and install morphizen-hipdnn and its dependencies on Windows platform

---

## 📋 Table of Contents

1. [Environment Information](#environment-information)
2. [Operation Steps](#operation-steps)
3. [Issues Encountered and Solutions](#issues-encountered-and-solutions)
4. [Build Results](#build-results)
5. [Environment Configuration Summary](#environment-configuration-summary)
6. [Appendix](#appendix)

---

## Environment Information

### Hardware Environment
- **GPU Model:** AMD Radeon(TM) 8050S Graphics
- **GPU Architecture:** gfx1151 (RDNA 3)
- **Compute Capability:** major: 11, minor: 5
- **Video Memory:** 24.26 GB (integrated graphics)
- **Multiprocessor Count:** 16
- **Clock Rate:** 2600 MHz

### Software Environment
- **Operating System:** Windows 10 (Build 26100)
- **CMake Version:** 3.29
- **Shell:** PowerShell
- **Compiler:** MSVC (Visual Studio)

### Directory Structure
```
D:\Users\mingyue\hipdnn\workspace\
├── MorphiZen/                    # MorphiZen main repository
├── morphizen-hipdnn/             # hipDNN integration project
├── onnxruntime/                  # ONNX Runtime source code
├── build/
│   ├── onnxruntime/              # ONNX Runtime build output
│   └── morphizen-hipdnn/         # morphizen-hipdnn build output
└── local/                        # Unified installation directory

D:\therock/                       # TheRock ROCm SDK
```

---

## Operation Steps

### Step 1: Install AMD GPU Drivers ✅

**Operations:**
1. Download driver from AMD official website
   - Download URL: https://www.amd.com/en/support/download/drivers.html
   - Select appropriate driver based on GPU model

2. Install driver and restart system

3. Verify installation
   - Open **Device Manager**
   - Navigate to **Display adapters**
   - Confirm display: `AMD Radeon(TM) 8050S Graphics`

**Result:** ✅ Driver installed successfully

---

### Step 2: Install ROCm SDK (TheRock) ✅

**Operations:**

**2.1 Download TheRock SDK**
- Source: https://therock-nightly-tarball.s3.amazonaws.com/index.html
- Select version corresponding to GPU architecture (gfx115X series)

**2.2 Extract to target directory**
```powershell
# Extract TheRock tarball to D:\therock
tar -xzf therock-dist-windows-*.tar.gz -C D:\
```

**Installation Location:** `D:\therock`

**Result:** ✅ TheRock SDK installed successfully

---

### Step 3: Obtain GPU Architecture Information ✅

**Command Executed:**
```powershell
D:\therock\bin\hipInfo.exe
```

**Key Output Information:**
```
Name:                             AMD Radeon(TM) 8050S Graphics
gcnArchName:                      gfx1151
major:                            11
minor:                            5
totalGlobalMem:                   24.26 GB
memInfo.free:                     24.10 GB (99%)
multiProcessorCount:              16
clockRate:                        2600 Mhz
isIntegrated:                     1
```

**Key Parameters:**
- **gcnArchName: gfx1151** (critical parameter for subsequent compilation and configuration)
- GFX Family: gfx115X (RDNA 3 architecture)

**Result:** ✅ Successfully obtained GPU architecture information

---

### Step 4: Switch Working Directory

**Command Executed:**
```powershell
cd D:\Users\mingyue\hipdnn\workspace
```

---

### Step 5: Build and Install ONNX Runtime ✅

**Operations:**

**5.1 Clone ONNX Runtime Repository**
```powershell
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

**5.2 Configure and Build ONNX Runtime**
```powershell
./build.bat --use_vitisai `
  --config Debug `
  --build_shared_lib `
  --parallel `
  --compile_no_warning_as_error `
  --skip_submodule_sync `
  --build_dir ../build/onnxruntime `
  --skip_tests `
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\Users\mingyue\hipdnn\workspace\local
```

**Build Parameters Explained:**
- `--use_vitisai` - Enable Vitis AI support (for custom operators)
- `--config Debug` - Debug mode compilation
- `--build_shared_lib` - Build shared library
- `--parallel` - Parallel compilation
- `--compile_no_warning_as_error` - Warnings not treated as errors
- `--skip_submodule_sync` - Skip submodule sync
- `--build_dir ../build/onnxruntime` - Build output directory
- `--skip_tests` - Skip tests
- `CMAKE_INSTALL_PREFIX` - Installation path

**5.3 Install to Specified Directory**
```powershell
cmake --build ../build/onnxruntime/Debug/ --target install
```

**Installation Location:** `D:\Users\mingyue\hipdnn\workspace\local`

**Result:** ✅ ONNX Runtime built and installed successfully

---

### Step 6: Clone MorphiZen Related Projects ✅

**Operations:**

**6.1 Return to Workspace Directory**
```powershell
cd D:\Users\mingyue\hipdnn\workspace
```

**6.2 Clone MorphiZen Repository**
```powershell
git clone git@gitenterprise.xilinx.com:VitisAI/MorphiZen.git --recursive
```
- Use `--recursive` parameter to clone all submodules

**6.3 Clone morphizen-hipdnn Repository**
```powershell
git clone git@gitenterprise.xilinx.com:VitisAI/morphizen-hipdnn.git
```

**Result:** ✅ Repositories cloned successfully

---

### Step 7: Configure morphizen-hipdnn Project

**7.1 Initial Configuration Command**
```powershell
cd morphizen-hipdnn
cmake -DBUILD_SHARED_LIBS=OFF `
  -B ../build/morphizen-hipdnn `
  -S . `
  -DCMAKE_INSTALL_PREFIX=D:\Users\mingyue\hipdnn\workspace\local `
  -DTHEROCK_DIST="D:\therock"
```

**CMake Configuration Parameters:**
- `-DBUILD_SHARED_LIBS=OFF` - Build static libraries
- `-B ../build/morphizen-hipdnn` - Build directory
- `-S .` - Source directory
- `-DCMAKE_INSTALL_PREFIX` - Installation prefix (same as ONNX Runtime)
- `-DTHEROCK_DIST="D:\therock"` - Specify TheRock HIP SDK path

**Initial Result:** ❌ Encountered configuration errors (see next section)

---

## Issues Encountered and Solutions

### Issue 1: nlohmann_json Package Not Found ❌

**Error Message:**
```
CMake Error: Could not find a package configuration file provided by "nlohmann_json"
```

**Root Cause:**
TheRock SDK's nlohmann_json CMake configuration file contains problematic `INTERFACE_SOURCES` attribute.

**Solution:** ✅

**File Path:** `D:\therock\share\cmake\nlohmann_json\nlohmann_jsonTargets.cmake`

**Modifications:**
1. Open file and find `set_target_properties(nlohmann_json::nlohmann_json ...)`
2. **Remove the `INTERFACE_SOURCES` line** (if it exists)

**Example Modification:**
```cmake
# Before:
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_SOURCES "B:/build/some/missing/file.cpp"  # ← Remove this line
)

# After:
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)
```

**Reference Documentation:** `doc/HIPDNNEP_INSTALLATION.md` lines 128-136

---

### Issue 2: hipdnn_frontend Contains Non-existent Path ❌

**Error Message:**
```
CMake Error: Imported target "hipdnn_frontend" includes non-existent path
  "B:/build/third-party/flatbuffers/dist/include"
```

**Root Cause:**
TheRock SDK's CMake configuration files contain hardcoded absolute paths from build time (`B:/build/...`), which do not exist on the current machine.

**Solution:** ✅

**Files to Modify:**

#### File 1: `D:\therock\lib\cmake\hipdnn_frontend\hipdnn_frontendTargets.cmake`

**Modifications:**
1. Find `set_target_properties(hipdnn_frontend ...)`
2. Locate the `INTERFACE_INCLUDE_DIRECTORIES` line
3. **Remove all absolute paths starting with `B:/build/`**
4. **Keep only `${_IMPORT_PREFIX}/include`**

**Example Modification:**
```cmake
# Before:
set_target_properties(hipdnn_frontend PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/flatbuffers/dist/include;B:/build/other/path;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
)

# After:
set_target_properties(hipdnn_frontend PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
)
```

#### File 2: `D:\therock\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake`

**Modifications:**
Similarly, remove all hardcoded absolute paths (`B:/build/...`), keep only `${_IMPORT_PREFIX}/include`

**Example Modification:**
```cmake
# Before:
INTERFACE_INCLUDE_DIRECTORIES "B:/build/third-party/llvm-project/install/include;B:/build/third-party/json/install/include;${_IMPORT_PREFIX}/include"

# After:
INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
```

**Reference Documentation:** `doc/HIPDNNEP_INSTALLATION.md` lines 102-137

---

### Issue 3: GSL Header File Not Found ❌

**Error Message:**
```
error C1083: Cannot open include file: 'gsl/gsl': No such file or directory
Location: D:\Users\mingyue\hipdnn\workspace\onnxruntime\onnxruntime\core\providers\vitisai\include\vaip\vaip_gsl.h(8,10)
```

**Root Cause:**
- Missing **Microsoft GSL** (Guidelines Support Library)
- ONNX Runtime's VitisAI provider header files depend on this library
- Error triggered when compiling MorphiZen's `mem_binary` tests

**Solution:** ✅

Modify MorphiZen project's CMakeLists.txt files to explicitly add Microsoft GSL dependency.

#### File 1: `MorphiZen/mem_binary/CMakeLists.txt`

**Modification (line 40):**
```diff
target_link_libraries(
  ${LIB_NAME}
  PRIVATE morphizen-utils
- PUBLIC ZLIB::ZLIB glog::glog)
+ PUBLIC ZLIB::ZLIB glog::glog Microsoft.GSL::GSL)
```

#### File 2: `MorphiZen/mem_binary/test/CMakeLists.txt`

**Modification (line 36):**
```diff
target_link_libraries(${TEST_EXE_NAME} PRIVATE GTest::gtest morphizen-utils
-                                               ZLIB::ZLIB glog::glog)
+                                               ZLIB::ZLIB glog::glog Microsoft.GSL::GSL)
```

**Modification Summary:**
- Added `Microsoft.GSL::GSL` dependency to `mem_binary` library
- Added `Microsoft.GSL::GSL` dependency to `test_mem_binary` test program
- This properly links the Microsoft GSL library and resolves header file issues

---

### Step 8: Successful Build and Installation ✅

**Command Executed:**
```powershell
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

**Build Parameters:**
- `--build ../build/morphizen-hipdnn` - Build directory
- `--config Debug` - Debug configuration mode
- `--target install` - Execute install target

**Result:** ✅ Build succeeded!

---

## Build Results

### Installation Configuration
- **Build Configuration:** Debug
- **Build Directory:** `D:\Users\mingyue\hipdnn\workspace\build\morphizen-hipdnn`
- **Installation Directory:** `D:\Users\mingyue\hipdnn\workspace\local`

### Installed Files

#### Executables (bin/)
```
D:\Users\mingyue\hipdnn\workspace\local\bin\
├── vaip_config.json                    # VAIP configuration file
├── morphizen-graph-opt.exe             # Graph optimization tool
├── morphizen-tar.exe                   # TAR packaging tool
├── morphizen-compile-pattern.exe       # Pattern compilation tool
├── morphizen-onnx-grep.exe             # ONNX search tool
├── hello_plugin_dll.exe                # Plugin example
└── hello_runner.exe                    # Runner example
```

#### Libraries (lib/)
```
D:\Users\mingyue\hipdnn\workspace\local\lib\
├── onnxruntime_vitisai_ep.lib          # ONNX Runtime VitisAI EP (static library)
├── onnxruntime_vitisai_ep.dll          # ONNX Runtime VitisAI EP (dynamic library)
├── hello_plugin_dll.lib                # Plugin library
└── [Other static libraries and protobuf files]
```

#### Component List
Built the following components:
- ✅ `hipdnn_proto` - Protobuf definitions
- ✅ `morphizen-level1-pass-hipdnn` - Level-1 Pass
- ✅ `morphizen-custom-op-hipdnn` - Custom Operator
- ✅ `morphizen-unit-tests` - Unit tests
- ✅ Various tools and example programs

---

## Environment Configuration Summary

### Key Path Configuration

| Component | Path/Value |
|-----------|------------|
| **GPU Model** | AMD Radeon(TM) 8050S Graphics |
| **GPU Architecture** | gfx1151 |
| **GFX Family** | gfx115X (RDNA 3) |
| **TheRock SDK** | D:\therock |
| **Workspace Directory** | D:\Users\mingyue\hipdnn\workspace |
| **Installation Directory** | D:\Users\mingyue\hipdnn\workspace\local |
| **Build Directory** | D:\Users\mingyue\hipdnn\workspace\build |
| **ONNX Runtime Source** | D:\Users\mingyue\hipdnn\workspace\onnxruntime |
| **MorphiZen Source** | D:\Users\mingyue\hipdnn\workspace\MorphiZen |
| **morphizen-hipdnn Source** | D:\Users\mingyue\hipdnn\workspace\morphizen-hipdnn |

### CMake Configuration Parameters (Complete)

```powershell
cmake -DBUILD_SHARED_LIBS=OFF `
  -B ../build/morphizen-hipdnn `
  -S . `
  -DCMAKE_INSTALL_PREFIX=D:\Users\mingyue\hipdnn\workspace\local `
  -DTHEROCK_DIST="D:\therock"
```

### Environment Variables (Recommended Setup)

```powershell
# TheRock SDK
$env:THEROCK_DIST = "D:\therock"
$env:HIP_PLATFORM = "amd"
$env:PATH = "D:\therock\bin;" + $env:PATH

# Installation directory
$env:PATH = "D:\Users\mingyue\hipdnn\workspace\local\bin;" + $env:PATH
```

---

## Appendix

### A. Completed Steps Checklist

- [x] 1. Install AMD GPU drivers
- [x] 2. Install ROCm SDK (TheRock)
- [x] 3. Obtain GPU architecture information (gfx1151)
- [x] 4. Switch working directory
- [x] 5. Build and install ONNX Runtime (with VitisAI)
- [x] 6. Clone MorphiZen and morphizen-hipdnn repositories
- [x] 7. Fix TheRock SDK CMake configuration files
  - [x] nlohmann_json configuration
  - [x] hipdnn_frontend configuration
  - [x] hipdnn_data_sdk configuration
- [x] 8. Fix GSL dependency issue
- [x] 9. Successfully build and install morphizen-hipdnn

### B. Modified Files List

#### TheRock SDK Configuration File Fixes
1. `D:\therock\share\cmake\nlohmann_json\nlohmann_jsonTargets.cmake`
   - Removed `INTERFACE_SOURCES` line

2. `D:\therock\lib\cmake\hipdnn_frontend\hipdnn_frontendTargets.cmake`
   - Removed hardcoded absolute paths (`B:/build/...`)

3. `D:\therock\lib\cmake\hipdnn_data_sdk\hipdnn_data_sdkTargets.cmake`
   - Removed hardcoded absolute paths (`B:/build/...`)

#### MorphiZen Source Code Modifications
1. `MorphiZen/mem_binary/CMakeLists.txt`
   - Added `Microsoft.GSL::GSL` dependency

2. `MorphiZen/mem_binary/test/CMakeLists.txt`
   - Added `Microsoft.GSL::GSL` dependency

### C. Reference Documentation

Related documentation in the project:
- `doc/HIPDNN_WINDOWS_SETUP.md` - hipDNN setup guide for Windows platform
- `doc/HIPDNNEP_INSTALLATION.md` - hipDNNEP installation guide
- `doc/BUILD_FIXES.md` - Build fixes documentation
- `doc/IMPLEMENTATION_GUIDE.md` - Implementation guide
- `doc/CONV_IMPLEMENTATION.md` - Convolution implementation documentation

### D. Next Steps Recommendations

1. **Run Tests for Verification**
   ```powershell
   cd D:\Users\mingyue\hipdnn\workspace\build\morphizen-hipdnn
   ctest -C Debug
   ```

2. **Configure Permanent Environment Variables**
   - Add TheRock SDK path to system PATH
   - Set HIP_PLATFORM=amd
   - Set THEROCK_DIST environment variable

3. **Run Example Programs**
   ```powershell
   D:\Users\mingyue\hipdnn\workspace\local\bin\hello_runner.exe
   ```

4. **Integration with Other Projects**
   - Use the installed ONNX Runtime VitisAI EP
   - Develop custom operators

### E. Troubleshooting Tips

If you encounter runtime errors:

1. **Check DLL Dependencies**
   ```powershell
   dumpbin /dependents D:\Users\mingyue\hipdnn\workspace\local\lib\onnxruntime_vitisai_ep.dll
   ```

2. **Verify HIP Runtime**
   ```powershell
   D:\therock\bin\hipconfig.exe --version
   ```

3. **Check GPU Visibility**
   ```powershell
   D:\therock\bin\hipInfo.exe
   ```

---

## Summary

This build successfully completed the following main tasks:
1. ✅ Configured AMD GPU drivers and ROCm SDK (TheRock)
2. ✅ Built and installed ONNX Runtime (with VitisAI support)
3. ✅ Resolved TheRock SDK configuration file path issues
4. ✅ Fixed GSL dependency issue
5. ✅ Successfully built morphizen-hipdnn and all components

All components have been installed to `D:\Users\mingyue\hipdnn\workspace\local` and are ready for use or further development.

---

**Document Generated:** January 12, 2026  
**Status:** ✅ Build completed successfully
