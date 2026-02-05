# Building onnx-hipdnn-ep

This document provides detailed instructions for building the onnx-hipdnn-ep project on Windows with Visual Studio.

## Prerequisites

### 1. Visual Studio 2022

Install Visual Studio with the "Desktop development with C++" workload:
- MSVC v143 or higher (19.44+)
- Windows SDK
- CMake tools for Windows

### 2. CMake 3.27+

Download from https://cmake.org/download/ or install via:
```powershell
winget install Kitware.CMake
```

### 3. Python 3.10+

Required for build scripts and protobuf:
```powershell
winget install Python.Python.3.12
```

### 4. Git

```powershell
winget install Git.Git
```

### 5. Ninja Build System

Required for faster builds:
```powershell
winget install Ninja-build.Ninja
```

### 6. TheRock ROCm SDK (Required)

For GPU acceleration with MIOpen and hipBLASLt, install TheRock:

1. Download nightly builds from: https://therock-nightly-tarball.s3.amazonaws.com/index.html
2. Extract to `C:\dist\therock` or `D:\Develop\m\dist\therock`
3. Set environment variable:
   ```powershell
   $env:THEROCK_DIST = "C:\dist\therock"
   ```

**Note:** TheRock is required for onnx-hipdnn-ep as it provides MIOpen and hipBLASLt libraries.

**Important:** Choose the TheRock build that matches your GPU architecture (e.g., `therock-dist-windows-gfx1151-*.tar.gz` for gfx1151 GPUs).

### 7. GPU Architecture (HIP_ARCHITECTURES)

The HIP kernels must be compiled for your specific GPU architecture. Using the wrong architecture will result in runtime crashes.

**Finding your GPU architecture:**
```powershell
# Using rocminfo from TheRock
& "$env:THEROCK_DIST\bin\rocminfo.exe" | Select-String "gfx"
```

**Common AMD GPU architectures:**

| Architecture | GPUs |
|--------------|------|
| `gfx1100` | Radeon RX 7900 XTX/XT (Navi 31) |
| `gfx1102` | Radeon RX 7600 (Navi 33) |
| `gfx1103` | Radeon 780M/760M iGPU (Phoenix) |
| `gfx1151` | Radeon PRO W7000 series |
| `gfx90a` | MI200 series (CDNA2) |
| `gfx942` | MI300 series (CDNA3) |

**Setting the architecture:**
```powershell
# Detect architecture
$env:HIP_ARCHITECTURES = & "$env:THEROCK_DIST\lib\llvm\bin\amdgpu-arch.exe"

# Pass to CMake
cmake -DHIP_ARCHITECTURES=$env:HIP_ARCHITECTURES <build_dir>
```

### 8. ONNX Runtime Source (Required)

ONNX Runtime source code is required for headers and API definitions:

```powershell
cd D:\Develop\m\source
git clone --recursive https://github.com/microsoft/onnxruntime.git
```

**Note:** You must build ONNX Runtime before building onnx-hipdnn-ep. See the "Building ONNX Runtime" section below.

## Directory Structure

The recommended directory structure is:

```
D:\Develop\m\              # Or C:\Develop\m\
├── source\
│   ├── onnx-hipdnn-ep\    # This project
│   ├── MorphiZen\         # MorphiZen framework (auto-fetched if not present)
│   └── onnxruntime\       # ONNX Runtime source (REQUIRED - clone and build first)
├── build\
│   ├── onnx-hipdnn-ep\    # onnx-hipdnn-ep build output
│   └── onnxruntime\       # ONNX Runtime build output
├── local\                 # Install prefix
└── dist\
    └── therock\           # TheRock SDK
```

**Note:** Adjust paths according to your installation location.

## Building ONNX Runtime (Required First)

Before building onnx-hipdnn-ep, you need to build ONNX Runtime first, as onnx-hipdnn-ep depends on ONNX Runtime headers and may require the built libraries for some components.

### Step 1: Clone ONNX Runtime (if not already cloned)

```cmd
cd D:\Develop\m\source
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

### Step 2: Build ONNX Runtime (Release Configuration)

For a Release build with VitisAI EP support (required for onnx-hipdnn-ep):

```cmd
REM Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Build ONNX Runtime with VitisAI EP in Release mode using Ninja (this will take 30-60 minutes)
.\build.bat --config Release ^
  --cmake_generator Ninja ^
  --use_vitisai ^
  --build_shared_lib ^
  --parallel ^
  --compile_no_warning_as_error ^
  --skip_submodule_sync ^
  --build_dir ..\build\onnxruntime ^
  --skip_tests ^
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\Develop\m\local
```

**Important Options:**
- `--config Release` - Build in Release mode (optimized, recommended for production)
- `--cmake_generator Ninja` - Use Ninja build system (faster than Visual Studio, recommended)
- `--use_vitisai` - Enable VitisAI Execution Provider (required for onnx-hipdnn-ep)
- `--build_shared_lib` - Build onnxruntime.dll
- `--parallel` - Enable parallel compilation
- `--compile_no_warning_as_error` - Don't treat warnings as errors
- `--skip_submodule_sync` - Skip submodule update (if already synced)
- `--build_dir` - Specify build output directory (relative to onnxruntime source)
- `--skip_tests` - Skip test building to save time
- `--cmake_extra_defines` - Pass additional CMake definitions

**Note:** The build directory path `..\build\onnxruntime` is relative to the onnxruntime source directory and will resolve to `D:\Develop\m\build\onnxruntime\Release`.

**Alternative Generator:** If you prefer Visual Studio generator, omit the `--cmake_generator Ninja` option. However, Ninja is faster and recommended.

**Debug Build:** If you need a debug build for development, use `--config Debug` instead of `--config Release`.

### Step 3: Install ONNX Runtime (Required)

After building ONNX Runtime, you must install it so that onnx-hipdnn-ep can find the CMake configuration files:

```cmd
cd D:\Develop\m\source\build\onnxruntime\Release
cmake --install . --prefix D:\Develop\m\local
```

This will install:
- Headers to `D:\Develop\m\local\include\onnxruntime\`
- Libraries to `D:\Develop\m\local\lib\`
- DLLs to `D:\Develop\m\local\bin\`
- **CMake config** to `D:\Develop\m\local\lib\cmake\onnxruntime\` (required for onnx-hipdnn-ep)

**Note:** This step is required before building onnx-hipdnn-ep.

### Alternative: Minimal Build

If you only need the headers and don't need a full ONNX Runtime build, you can do a minimal configuration:

```cmd
cd D:\Develop\m\source\onnxruntime
mkdir -p build\minimal
cd build\minimal
cmake ..\..
```

This will generate the necessary configuration files without a full build.

## Building onnx-hipdnn-ep

**Prerequisites:** Ensure you have built ONNX Runtime first (see "Building ONNX Runtime" section above).

## Build Steps

### Step 1: Set Up Environment

Open a **Developer Command Prompt for VS 2022** or run:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

### Step 2: Set TheRock Path

```cmd
set THEROCK_DIST=C:\dist\therock
set HIP_PLATFORM=amd
```

### Step 3: Configure with CMake

```cmd
cmake -G "Ninja" ^
  -DCMAKE_CXX_FLAGS="/EHsc" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  -B D:\Develop\m\build\onnx-hipdnn-ep ^
  -S . ^
  -DCMAKE_INSTALL_PREFIX=D:\Develop\m\local ^
  -DCMAKE_PREFIX_PATH=D:\Develop\m\local ^
  -DTHEROCK_DIST=%THEROCK_DIST% ^
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON ^
  -Dmorphizen_ENABLE_UNIT_TEST=OFF
```

**Important CMake Options:**
- `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` - Uses dynamic runtime (/MD) to match TheRock's protobuf
- `-Dmorphizen_ENABLE_ORT_BRIDGE=ON` - Enables the new ORT API 2.0 support
- `-Dmorphizen_ENABLE_UNIT_TEST=OFF` - Disables unit tests to avoid ONNX Runtime dependency

### Step 4: Build

```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep
```

### Step 5: Install (Optional)

```cmd
cmake --install D:\Develop\m\build\onnx-hipdnn-ep
```

## Build Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: `Debug`, `Release`, `RelWithDebInfo` |
| `BUILD_SHARED_LIBS` | `OFF` | Build shared libraries |
| `THEROCK_DIST` | `` | Path to TheRock ROCm SDK (required) |
| `CMAKE_MSVC_RUNTIME_LIBRARY` | `MultiThreadedDLL` | MSVC runtime: `/MD` for Release, `/MDd` for Debug |
| `morphizen_ENABLE_ORT_BRIDGE` | `ON` | Enable ONNX Runtime 2.0 bridge |
| `morphizen_ENABLE_UNIT_TEST` | `OFF` | Enable unit tests (requires ONNX Runtime) |

### Debug Build

For a debug build with debugging symbols:

```cmd
cmake -G "Ninja" ^
  -DCMAKE_CXX_FLAGS="/EHsc" ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebugDLL ^
  -B D:\Develop\m\build\onnx-hipdnn-ep ^
  -S . ^
  -DCMAKE_INSTALL_PREFIX=D:\Develop\m\local ^
  -DCMAKE_PREFIX_PATH=D:\Develop\m\local ^
  -DTHEROCK_DIST=%THEROCK_DIST% ^
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON ^
  -Dmorphizen_ENABLE_UNIT_TEST=OFF
```

## Build Outputs

After a successful build, you'll find:

### Libraries

Key libraries in `D:\Develop\m\build\onnx-hipdnn-ep\`:

| Library | Description |
|---------|-------------|
| `onnxruntime_vitisai_ep.dll` | MorphiZen VitisAI EP dynamic library |
| `morphizen-custom-op-rocm.lib` | ROCm custom operator (MIOpen + hipBLASLt) |
| `vaip-pass_level1_rocm.lib` | Level-1 orchestration pass |
| `vaip-pass_level2_rocm_conv.lib` | Level-2 Conv pattern matching pass |
| `vaip-pass_level2_rocm_gemm.lib` | Level-2 Gemm pattern matching pass |
| `rocm_proto.lib` | Protocol buffer definitions |

### Pattern Files

Located in `patterns/`:

| File | Description |
|------|-------------|
| `conv.json` | Conv pattern definitions for MIOpen |
| `gemm.json` | Gemm pattern definitions for hipBLASLt |

### Configuration Files

Located in `etc/`:

| File | Description |
|------|-------------|
| `morphizen_config.json` | Morphizen EP pass configuration |

## Troubleshooting

### Error: "TheRock ROCm SDK not found"

The build script checks multiple locations:
1. `C:\Develop\m\dist\therock`
2. `C:\dist\therock`
3. `C:\Develop\TheRock`

Ensure TheRock is installed in one of these locations or set `THEROCK_DIST` manually.

### Error: "Visual Studio 2022 not found"

Ensure Visual Studio 2022 is installed and run vcvarsall.bat to set up the environment:
```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

### Error: "Could NOT find Boost" or "Could NOT find onnxruntime"

This occurs if MorphiZen tries to build tools (graph-opt, tar, pattern-gen, onnx-grep) that require these dependencies.

You may need to patch MorphiZen's CMakeLists.txt to skip these tools:
```powershell
$content = Get-Content '..\MorphiZen\CMakeLists.txt'
$content = $content -replace 'add_subdirectory\(graph-opt\)', '#add_subdirectory(graph-opt)'
$content = $content -replace 'add_subdirectory\(tar\)', '#add_subdirectory(tar)'
$content = $content -replace 'add_subdirectory\(pattern-gen\)', '#add_subdirectory(pattern-gen)'
$content = $content -replace 'add_subdirectory\(onnx-grep\)', '#add_subdirectory(onnx-grep)'
Set-Content '..\MorphiZen\CMakeLists.txt' $content
```

### Error: "HIP is required but not found"

Ensure TheRock is installed and `THEROCK_DIST` is set correctly. Check that:
```cmd
dir %THEROCK_DIST%\lib\cmake\hip
dir %THEROCK_DIST%\lib\cmake\miopen
dir %THEROCK_DIST%\lib\cmake\hipblaslt
```

All three should exist.

### Error: "LNK2038: mismatch detected for 'RuntimeLibrary'"

Ensure `CMAKE_MSVC_RUNTIME_LIBRARY` matches across all dependencies:
- Release builds: Use `MultiThreadedDLL` (/MD)
- Debug builds: Use `MultiThreadedDebugDLL` (/MDd)

Note: onnx-hipdnn-ep uses **dynamic runtime** (/MD) to match TheRock's protobuf library, unlike morphizen-hipblaslt which uses static runtime (/MT).

### Error: "CMake source path mismatch"

This occurs when the build directory has a cached configuration from a different source location. Clean the build directory:
```cmd
rmdir /s /q D:\Develop\m\build\onnx-hipdnn-ep
```

Then reconfigure.

### Build is Slow

- Use Ninja generator: `cmake -G Ninja ...`
- Use `-j N` to specify parallel jobs: `cmake --build . -j 16`
- Use an SSD for the build directory
- Close other applications to free up RAM for parallel compilation

### Warning: "Fetching RAI version failed, using default: 9999.0.0"

This is a benign warning. The build will continue successfully. It occurs because the version fetching script expects a specific git structure.

### Error: "ONNX Runtime headers not found"

Ensure ONNX Runtime is cloned and located at the correct path:
```cmd
dir D:\Develop\m\source\onnxruntime\onnxruntime\core\providers\vitisai\include
```

If the directory doesn't exist, clone ONNX Runtime:
```cmd
cd D:\Develop\m\source
git clone --recursive https://github.com/microsoft/onnxruntime.git
```

### Error: "Cannot find onnxruntimeConfig.cmake"

This error may occur if you're trying to build components that require a fully built ONNX Runtime. Solutions:

1. **Build ONNX Runtime first** (recommended):
   ```cmd
   cd D:\Develop\m\source\onnxruntime
   .\build.bat --config Release --cmake_generator Ninja --use_vitisai --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ..\build\onnxruntime --skip_tests
   ```

2. **Patch MorphiZen CMakeLists.txt**: Disable components that require ONNX Runtime's CMake config (graph-opt, tar, pattern-gen, onnx-grep).

3. **Set ONNX Runtime install path**:
   ```cmd
   set CMAKE_PREFIX_PATH=D:\Develop\m\local;D:\Develop\m\build\onnxruntime\Release
   ```

## Running Tests

### Integration Tests

After building, you can run integration tests with the VitisAI EP:

```cmd
cd D:\Develop\m\build\onnx-hipdnn-ep\bin
set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1
set THEROCK_DIST=C:\dist\therock
set PATH=%THEROCK_DIST%\bin;%PATH%
.\ort_integration_test.exe
```

### Environment Variables for Testing

| Variable | Purpose |
|----------|---------|
| `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1` | Required for ORT 2.0 V2 device API |
| `MORPHIZEN_DEBUG_ROCM=2` | Enable debug logging (optional) |
| `GLOG_logtostderr=1` | Enable glog stderr output (optional) |
| `GLOG_minloglevel=0` | Show all glog messages (optional) |

### Test Models

Test models can be generated using the `gen_sample_model.py` script or use the pre-built models in the `test/models/` directory.

## Development Tips

### Incremental Build

After the first successful build, subsequent builds are much faster:
```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep --config Release
```

### Force Reconfiguration

To force CMake reconfiguration, delete the cache and reconfigure:
```cmd
del D:\Develop\m\build\onnx-hipdnn-ep\CMakeCache.txt
cmake -G Ninja -B D:\Develop\m\build\onnx-hipdnn-ep -S . [options...]
```

### Rebuild Single Target

```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep --target morphizen-custom-op-rocm
```

### Clean Build

```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep --target clean
```

### Enable Verbose Output

```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep -v
```

### View Build Targets

```cmd
cmake --build D:\Develop\m\build\onnx-hipdnn-ep --target help
```

## Continuous Integration

For automated builds, use the following command sequence:

```cmd
REM Set environment
set THEROCK_DIST=C:\dist\therock
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Clean build
rmdir /s /q D:\Develop\m\build\onnx-hipdnn-ep

REM Configure and build
cd D:\Develop\m\source\onnx-hipdnn-ep
cmake -G Ninja -B ..\build\onnx-hipdnn-ep -S . -DTHEROCK_DIST=%THEROCK_DIST% -DHIP_ARCHITECTURES=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build ..\build\onnx-hipdnn-ep --config Release
```

## Next Steps

After successfully building onnx-hipdnn-ep:

1. Review [01_DESIGN.md](01_DESIGN.md) for architecture overview
2. See [02_LEVEL1_PASS_DESIGN.md](02_LEVEL1_PASS_DESIGN.md) for Level-1 pass design
3. Check [06_LOGGING.md](06_LOGGING.md) for debugging capabilities
4. Refer to [09_TROUBLESHOOTING.md](09_TROUBLESHOOTING.md) for runtime issues

## See Also

- [01_DESIGN.md](01_DESIGN.md) - Architecture and design overview
- [02_LEVEL1_PASS_DESIGN.md](02_LEVEL1_PASS_DESIGN.md) - Level-1 pass implementation
- [06_LOGGING.md](06_LOGGING.md) - Logging system
- [09_TROUBLESHOOTING.md](09_TROUBLESHOOTING.md) - Runtime troubleshooting
- [README.md](../README.md) - Project overview
