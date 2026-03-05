<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Building & Testing Guide

This guide covers the **pre-built flow**: downloading ready-made LLVM/MLIR/LLD,
FlatBuffers, and Protobuf binaries instead of compiling them from source.

## Prerequisites

| Tool | Purpose |
|------|---------|
| **CMake** 3.20+ | Build system |
| **Ninja** | Build generator (faster than MSVC, required) |
| **MSVC 2022** | C++ compiler (Visual Studio Build Tools or full IDE) |
| **Python 3** | LLVM/MLIR tools runtime |
| **sccache** | Compiler cache (significantly speeds up rebuilds) |
| **`gh` CLI** | Downloading pre-built binaries (`gh auth login` required) |
| **`unzip`** | Extracting downloaded archives (available in Git Bash / MSYS2) |

**MSVC Environment Setup:**
- Launch git-bash from "Developer Command Prompt for VS XXXX" (where XXXX is your VS version: 2019, 2022, 2026, etc.) to ensure MSVC environment is available
- Or launch Claude Code from Developer Command Prompt

**sccache installation:**
```bash
# Windows (winget)
winget install Mozilla.sccache
# or via cargo
cargo install sccache
```

**Ninja installation:**
```bash
# Windows (winget)
winget install Ninja-build.Ninja
# or via pip
pip install ninja
```

## Directory Layout

All paths are relative to the directory that contains the project source
(`<workspace>/`). Clone the repository so that the source, build, and
prebuilt-local directories are siblings:

```
<workspace>/
├── <repo>/             # project source — this is $PWD for all commands below
├── build/
│   └── <repo>/         # cmake build output  ../build/$(basename $PWD)
└── prebuilt-local/     # pre-built Release binaries  ../prebuilt-local
    ├── bin/
    ├── include/
    └── lib/cmake/{llvm,mlir,lld,flatbuffers,protobuf,...}
```

## One-Time Setup

### 1. Download Pre-built Dependencies

Run from the project root (inside Git Bash / MSYS2):

```bash
bash scripts/setup-prebuilt.sh
```

This downloads the following assets from
`https://github.com/wcy123/llvm-mlir-prebuilt/releases` and extracts them into
`../prebuilt-local/`:

| Release tag | Asset |
|---|---|
| `llvm-22.1.0-release` | `llvm-22.1.0-release-windows-x64.zip` |
| `protobuf-34.0-release` | `protobuf-34.0-release-windows-x64.zip` |
| `flatbuffers-25.12.19-release` | `flatbuffers-25.12.19-release-windows-x64.zip` |

Already-downloaded zip files are skipped on subsequent runs.

### 2. Build ONNX Runtime (Required for BUILD_EP=ON)

If you plan to build with `-DBUILD_EP=ON` (to build the MorphiZen Execution Provider), you must first build and install ONNX Runtime.

**Clone ONNX Runtime** (if not already cloned):

```bash
cd ..  # Go to workspace directory (parent of project root)
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

**Build and install ONNX Runtime**:

```bash
# Build ONNX Runtime using build.bat (do NOT set CMAKE_INSTALL_PREFIX during build)
# This ensures ONNX Runtime uses its own FlatBuffers version, avoiding conflicts with prebuilt FlatBuffers
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --disable_memleak_checker

# Install to prebuilt-local (set prefix at install time, not during configuration)
PREBUILT_DIR=$(cd ../prebuilt-local && pwd)
cmake --install ../build/onnxruntime/Release --prefix "$PREBUILT_DIR"
```

**Verify installation**:

```bash
ls $PREBUILT_DIR/lib/cmake/onnxruntime/
# Should see onnxruntime-config.cmake and related files
```

**Note**: For standalone compiler builds (without the Execution Provider), you can skip this step. The project builds successfully without `-DBUILD_EP=ON`.

## Configure

Run from the project root (the cloned repository directory):

```bash
PREBUILT_DIR=$(cd ../prebuilt-local && pwd)
cmake -S . -B ../build/$(basename $PWD) \
  -G Ninja \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$PREBUILT_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DBUILD_MOCK_RUNTIME=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON \
  --fresh
```

**Key options:**

| Option | Value | Notes |
|--------|-------|-------|
| `-G Ninja` | — | Required; MSVC generator is not supported for this flow |
| `CMAKE_MSVC_RUNTIME_LIBRARY` | `MultiThreaded` | Must match pre-built binaries (`/MT`) |
| `CMAKE_BUILD_TYPE` | `Release` | Pre-built binaries are Release; Debug CRT (`/MTd`) is incompatible |
| `CMAKE_C/CXX_COMPILER_LAUNCHER` | `sccache` | Omit if sccache is not installed |
| `BUILD_MOCK_RUNTIME` | `ON` | Enables build without a GPU / ROCm installation |
| `ONNX_HIP_INCLUDE_LIT_TESTS` | `ON` | Includes the LIT test suite |

## Build

```bash
cmake --build ../build/$(basename $PWD) --config Release --parallel
```

## Test

```bash
ctest --test-dir ../build/$(basename $PWD) --config Release --verbose
```

To run only the LIT tests directly (with progress output):

```bash
cd ../build/$(basename $PWD)/test/lit
lit -sv .
```

## ABI Note

The pre-built binaries are compiled with the **Release `/MT`** runtime
(`MultiThreaded` static CRT). Using `-DCMAKE_BUILD_TYPE=Debug` or
`-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug` will produce runtime library
mismatch linker errors.

## Updating Pre-built Binaries

When new releases are published to `wcy123/llvm-mlir-prebuilt`, update the tag
variables at the top of `scripts/setup-prebuilt.sh` and re-run it. Delete the
old zip files in `../prebuilt-local/` if you want a clean re-download.

## Troubleshooting

### CMake cannot find LLVM/MLIR

**Symptom:** `Could not find package LLVM` or similar.

**Solution:** Verify `../prebuilt-local/lib/cmake/llvm/` exists. Re-run
`bash scripts/setup-prebuilt.sh` if the directory is missing.

### `gh` authentication error during setup

**Symptom:** `gh release download` fails with `HTTP 401`.

**Solution:** Run `gh auth login` and authenticate with a GitHub account that
has access to `wcy123/llvm-mlir-prebuilt`.

### sccache not found

**Symptom:** CMake configure error: `Could not find compiler launcher sccache`.

**Solution:** Install sccache (see Prerequisites) or remove the
`CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` options from the
configure command.

### Runtime library mismatch

**Symptom:** Linker errors mentioning `/MT` vs `/MTd` or `/MD`.

**Solution:** Use exactly `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` and
`-DCMAKE_BUILD_TYPE=Release`. Do not mix Debug and Release configurations.

### LIT tests not running (ctest reports instant pass)

**Symptom:** `MorphizenMLIRLitTests` completes in under 1 second and reports
`Passed` without printing any test names.

**Solution:** Ensure `lit` is installed via pip (`pip install lit`) and that
CMake found the correct `lit.exe` during configure. Re-run configure with
`--fresh` after installing lit.

### Missing DIA SDK library (diaguids.lib)

**Symptom:** `ninja: error: 'C:/msvsn2022/DIA SDK/lib/amd64/diaguids.lib' missing`

**Cause:** The prebuilt LLVM package has `C:\msvsn2022` hardcoded in LLVMExports.cmake for the DIA SDK path. This is a known LLVM issue: https://github.com/llvm/llvm-project/issues/111829

**Solution:** Create a junction from `C:\msvsn2022` to your Visual Studio installation:

```bash
# Auto-detect VS installation path and create junction
for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do mklink /J C:\msvsn2022 "%i"
```
