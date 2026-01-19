# Building morphizen-rocm

This document provides detailed instructions for building the morphizen-rocm project on Windows with Visual Studio.

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

**Note:** TheRock is required for morphizen-rocm as it provides MIOpen and hipBLASLt libraries.

### 7. ONNX Runtime Source (Required)

ONNX Runtime source code is required for headers and API definitions:

```powershell
cd D:\Develop\m\source
git clone --recursive https://github.com/microsoft/onnxruntime.git
```

**Note:** You must build ONNX Runtime before building morphizen-rocm. See the "Building ONNX Runtime" section below.

## Directory Structure

The recommended directory structure is:

```
D:\Develop\m\              # Or C:\Develop\m\
├── source\
│   ├── morphizen-rocm\    # This project
│   ├── MorphiZen\         # MorphiZen framework (auto-fetched if not present)
│   └── onnxruntime\       # ONNX Runtime source (REQUIRED - clone and build first)
├── build\
│   ├── morphizen-rocm\    # morphizen-rocm build output
│   └── onnxruntime\       # ONNX Runtime build output
├── local\                 # Install prefix
└── dist\
    └── therock\           # TheRock SDK
```

**Note:** The build script automatically detects whether you're on C: or D: drive and adjusts paths accordingly.

## Building ONNX Runtime (Required First)

Before building morphizen-rocm, you need to build ONNX Runtime first, as morphizen-rocm depends on ONNX Runtime headers and may require the built libraries for some components.

### Step 1: Clone ONNX Runtime (if not already cloned)

```cmd
cd D:\Develop\m\source
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

### Step 2: Build ONNX Runtime (Release Configuration)

For a Release build with VitisAI EP support (required for morphizen-rocm):

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
- `--use_vitisai` - Enable VitisAI Execution Provider (required for morphizen-rocm)
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

After building ONNX Runtime, you must install it so that morphizen-rocm can find the CMake configuration files:

```cmd
cd D:\Develop\m\source\build\onnxruntime\Release
cmake --install . --prefix D:\Develop\m\local
```

This will install:
- Headers to `D:\Develop\m\local\include\onnxruntime\`
- Libraries to `D:\Develop\m\local\lib\`
- DLLs to `D:\Develop\m\local\bin\`
- **CMake config** to `D:\Develop\m\local\lib\cmake\onnxruntime\` (required for morphizen-rocm)

**Note:** This step is required before building morphizen-rocm.

### Alternative: Minimal Build

If you only need the headers and don't need a full ONNX Runtime build, you can do a minimal configuration:

```cmd
cd D:\Develop\m\source\onnxruntime
mkdir -p build\minimal
cd build\minimal
cmake ..\..
```

This will generate the necessary configuration files without a full build.

## Quick Build (Windows)

**Prerequisites:** Ensure you have built ONNX Runtime first (see "Building ONNX Runtime" section above).

The simplest way to build is using the provided `build.bat` script:

```cmd
cd D:\morphizen-rocm\source\morphizen-rocm
build.bat
```

This script will:
1. Detect the current drive (C: or D:) and set paths accordingly
2. Set up TheRock environment (checking multiple common locations)
3. Initialize MSVC environment (supports both VS 2022 and VS 18 paths)
4. Patch MorphiZen to skip tools that require ONNX Runtime
5. Configure with CMake using Ninja generator
6. Build all targets

## Manual Build

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
  -B D:\Develop\m\build\morphizen-rocm ^
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
cmake --build D:\Develop\m\build\morphizen-rocm
```

### Step 5: Install (Optional)

```cmd
cmake --install D:\Develop\m\build\morphizen-rocm
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
  -B D:\Develop\m\build\morphizen-rocm ^
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

Key libraries in `D:\Develop\m\build\morphizen-rocm\`:

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
| `vaip_config.json` | VitisAI EP configuration |
| `vaip_config_disk.json` | Disk-based configuration variant |

## Troubleshooting

### Error: "TheRock ROCm SDK not found"

The build script checks multiple locations:
1. `C:\Develop\m\dist\therock`
2. `C:\dist\therock`
3. `C:\Develop\TheRock`

Ensure TheRock is installed in one of these locations or set `THEROCK_DIST` manually.

### Error: "Visual Studio 2022 not found"

The build script checks:
1. `C:\Program Files\Microsoft Visual Studio\2022\Community\`
2. `C:\Program Files\Microsoft Visual Studio\18\Community\`

If your Visual Studio is in a different location, modify `build.bat` accordingly.

### Error: "Could NOT find Boost" or "Could NOT find onnxruntime"

This occurs if MorphiZen tries to build tools (graph-opt, tar, pattern-gen, onnx-grep) that require these dependencies. The `build.bat` script automatically patches MorphiZen's CMakeLists.txt to skip these tools.

If building manually, ensure you add the patch step:
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

Note: morphizen-rocm uses **dynamic runtime** (/MD) to match TheRock's protobuf library, unlike morphizen-hipblaslt which uses static runtime (/MT).

### Error: "CMake source path mismatch"

This occurs when the build directory has a cached configuration from a different source location. Clean the build directory:
```cmd
rmdir /s /q D:\Develop\m\build\morphizen-rocm
```

Then reconfigure.

### Build is Slow

- Use Ninja generator (already default in build.bat)
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

2. **Use the build script's automatic patching**: The `build.bat` script automatically disables components that require ONNX Runtime's CMake config (graph-opt, tar, pattern-gen, onnx-grep).

3. **Set ONNX Runtime install path**:
   ```cmd
   set CMAKE_PREFIX_PATH=D:\Develop\m\local;D:\Develop\m\build\onnxruntime\Release
   ```

## Running Tests

### Integration Tests

After building, you can run integration tests with the VitisAI EP:

```cmd
cd D:\Develop\m\build\morphizen-rocm\bin
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

Generate test models using the provided Python scripts:

```cmd
cd D:\morphizen-rocm\source\morphizen-rocm\test
python gen_conv_model.py
python gen_gemm_model.py
```

## Development Tips

### Incremental Build

After the first successful build, subsequent builds are much faster:
```cmd
cd D:\morphizen-rocm\source\morphizen-rocm
.\build.bat
```

The script skips CMake configuration if `build.ninja` already exists.

### Force Reconfiguration

To force CMake reconfiguration:
```cmd
del D:\Develop\m\build\morphizen-rocm\build.ninja
.\build.bat
```

### Rebuild Single Target

```cmd
cmake --build D:\Develop\m\build\morphizen-rocm --target morphizen-custom-op-rocm
```

### Clean Build

```cmd
cmake --build D:\Develop\m\build\morphizen-rocm --target clean
```

### Enable Verbose Output

```cmd
cmake --build D:\Develop\m\build\morphizen-rocm -v
```

### View Build Targets

```cmd
cmake --build D:\Develop\m\build\morphizen-rocm --target help
```

## Continuous Integration

For automated builds, use the following command sequence:

```cmd
REM Set environment
set THEROCK_DIST=C:\dist\therock
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Clean build
rmdir /s /q D:\Develop\m\build\morphizen-rocm

REM Configure and build
cd D:\morphizen-rocm\source\morphizen-rocm
.\build.bat
```

## Next Steps

After successfully building morphizen-rocm:

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
