# Compiler Guide — Build, Test & Run

## Directory Layout

```
C:\Users\Administrator\shoucair\onnx_hipdnn_workspace\   (%WORKSPACE%)
├── depends\                       # Dependency sources
│   ├── llvm-project\              #   LLVM/MLIR/Clang/LLD
│   ├── protobuf\                  #   Protocol Buffers
│   ├── googletest\                #   Google Test
│   ├── glog\                      #   Google Logging
│   └── GSL\                       #   Microsoft GSL
├── onnxruntime\                   # ONNX Runtime source
├── hip-compiler\                 # MLIR-based ONNX→DLL compiler
├── onnx-hipdnn-ep\                # ORT Execution Provider (MorphiZen EP)
├── test_onnx_runner\              # Standalone ORT test harness
├── therock\                       # TheRock ROCm SDK (prebuilt)
│   └── bin\                       # ROCm runtime DLLs
├── local\                         # Install prefix (all projects install here)
│   ├── bin\                       # Executables + DLLs + config JSONs
│   ├── lib\                       # Static/import libs
│   └── include\                   # Headers
└── build\                         # Build output dirs
```

## Common Settings

All projects share these CMake settings on Windows. Set these variables once per terminal session:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set WORKSPACE=C:\Users\Administrator\shoucair\onnx_hipdnn_workspace
set LOCAL=%WORKSPACE%\local
set THEROCK=%WORKSPACE%\therock
```

| Setting | Value |
|---------|-------|
| Generator | Ninja |
| Build type | Release |
| MSVC runtime | `/MT` (static CRT) |
| Install prefix | `%LOCAL%` (`local\`) |
| CMAKE_PREFIX_PATH | `%LOCAL%` (`local\`) |
| Executables & DLLs | `%LOCAL%\bin\` |

---

## Dependencies

All dependencies are built and installed to `%LOCAL%`. Build them in the order listed below. Sources are cloned into `%WORKSPACE%\depends\`.

### Dep 1: protobuf (v21.12)

```bat
:: Clone (one-time)
git clone --branch v21.12 --depth 1 https://github.com/protocolbuffers/protobuf.git %WORKSPACE%\depends\protobuf

:: Configure and build
cmake -G Ninja ^
  -S %WORKSPACE%\depends\protobuf ^
  -B C:\b\protobuf ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -Dprotobuf_BUILD_TESTS=OFF ^
  -Dprotobuf_MSVC_STATIC_RUNTIME=ON

cmake --build C:\b\protobuf --config Release --parallel
cmake --install C:\b\protobuf --config Release
```

### Dep 2: gtest (Google Test v1.15.0)

```bat
:: Clone (one-time)
git clone --branch v1.15.0 --depth 1 https://github.com/google/googletest.git %WORKSPACE%\depends\googletest

:: Configure and build
cmake -G Ninja ^
  -S %WORKSPACE%\depends\googletest ^
  -B C:\b\googletest ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -DBUILD_GMOCK=ON

cmake --build C:\b\googletest --config Release --parallel
cmake --install C:\b\googletest --config Release
```

### Dep 3: glog (Google Logging v0.7.1)

**CRITICAL**: glog MUST be built as a **static library** with `-DBUILD_SHARED_LIBS=OFF`.

```bat
:: Clone (one-time)
git clone --branch v0.7.1 --depth 1 https://github.com/google/glog.git %WORKSPACE%\depends\glog

:: Configure and build
cmake -G Ninja ^
  -S %WORKSPACE%\depends\glog ^
  -B C:\b\glog ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DBUILD_TESTING=OFF

cmake --build C:\b\glog --config Release --parallel
cmake --install C:\b\glog --config Release
```

**Verification**: After install, `%LOCAL%\lib\glog.lib` should exist (static). If you see `glog.dll`, rebuild with `-DBUILD_SHARED_LIBS=OFF`.

### Dep 4: gsl (Microsoft GSL v4.0.0)

Header-only library — no build step needed, just configure and install.

```bat
:: Clone (one-time)
git clone --branch v4.0.0 --depth 1 https://github.com/microsoft/GSL.git %WORKSPACE%\depends\GSL

:: Configure and install
cmake -G Ninja ^
  -S %WORKSPACE%\depends\GSL ^
  -B C:\b\GSL ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -DGSL_TEST=OFF

cmake --install C:\b\GSL --config Release
```

### Dep 5: LLVM/MLIR/Clang/LLD

**CRITICAL**: LLVM must be built with matching runtime library, clang, LLD, and test utilities.

- **Source**: `%WORKSPACE%\depends\llvm-project`
- **Build dir**: `C:\b\llvm`
- **Build time**: 30–60 minutes, 20 GB+ disk space

```bat
:: Clone (one-time)
git clone --depth 1 https://github.com/llvm/llvm-project.git %WORKSPACE%\depends\llvm-project

:: Configure
cmake -G Ninja ^
  -S %WORKSPACE%\depends\llvm-project\llvm ^
  -B C:\b\llvm ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLVM_ENABLE_PROJECTS="mlir;lld;clang" ^
  -DLLVM_TARGETS_TO_BUILD=host ^
  -DLLVM_ENABLE_ASSERTIONS=ON ^
  -DLLVM_ENABLE_RTTI=OFF ^
  -DLLVM_INSTALL_UTILS=ON ^
  -DLLVM_INCLUDE_TESTS=ON ^
  -DLLVM_ENABLE_ZLIB=OFF ^
  -DLLVM_ENABLE_ZSTD=OFF ^
  --fresh

:: Build (try parallel first)
cmake --build C:\b\llvm --config Release --parallel

:: If C1041 PDB conflicts occur, fall back to serial linking then resume parallel
cmake --build C:\b\llvm --config Release --parallel 1
cmake --build C:\b\llvm --config Release --parallel

:: Install
cmake --install C:\b\llvm --config Release
```

| Setting | Why |
|---------|-----|
| `LLVM_ENABLE_PROJECTS="mlir;lld;clang"` | MLIR dialects, LLD linker, Clang compiler — all required by hip-compiler |
| `LLVM_INSTALL_UTILS=ON` | Installs `FileCheck`, `lit`, `count`, `not`, `split-file`, `llvm-dis` |
| `LLVM_INCLUDE_TESTS=ON` | Required for test utilities to be built |
| `BUILD_SHARED_LIBS=OFF` | Static libraries for consistent linking |
| `LLVM_ENABLE_ZLIB=OFF` / `ZSTD=OFF` | Avoids external compression library dependencies |
| `LLVM_ENABLE_ASSERTIONS=ON` | Useful for development/debugging |

### Dependency Verification

After building all dependencies, verify CMake config files exist:

```bat
dir %LOCAL%\lib\cmake\protobuf\protobuf-config.cmake
dir %LOCAL%\lib\cmake\GTest\GTestConfig.cmake
dir %LOCAL%\lib\cmake\glog\glog-config.cmake
dir %LOCAL%\lib\cmake\Microsoft.GSL\Microsoft.GSLConfig.cmake
dir %LOCAL%\lib\cmake\mlir\MLIRConfig.cmake
dir %LOCAL%\lib\cmake\llvm\LLVMConfig.cmake
```

---

## Step 1: ONNX Runtime

ORT is needed by onnx-hipdnn-ep (EP plugin headers/libs) and test_onnx_runner (test harness).

- **Source**: `%WORKSPACE%\onnxruntime`
- **Build dir**: `%WORKSPACE%\build\onnxruntime`

It is recommended to use **Git Bash** for the ORT build. The `build.bat` script has only been tested in Git Bash.

### Clone (one-time)

```bash
cd $WORKSPACE
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

### Build & Install (Git Bash)

```bash
./build.bat --config Release --build_shared_lib --parallel \
  --compile_no_warning_as_error --skip_submodule_sync \
  --build_dir ../build/onnxruntime --skip_tests \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local \
  --disable_memleak_checker

cmake --build ../build/onnxruntime/Release/ --target install
```

Installs `onnxruntime.dll`, headers, and import lib to `%LOCAL%`.

---

## Step 2: onnx-hipdnn-ep (MorphiZen EP)

- **Source**: `%WORKSPACE%\onnx-hipdnn-ep`
- **Build dir**: `%WORKSPACE%\build`

### Configure (fresh)

```bat
cmake -G Ninja ^
  -S %WORKSPACE%\onnx-hipdnn-ep ^
  -B %WORKSPACE%\build ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_PREFIX_PATH=%LOCAL% ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  -DTHEROCK_DIST=%THEROCK% ^
  -DHIP_PLATFORM=amd ^
  -DHIP_ARCHITECTURES=gfx1150
```

### Build

```bat
cmake --build %WORKSPACE%\build --config Release --parallel
```

### Install

```bat
cmake --install %WORKSPACE%\build --config Release
```

### What gets installed

| File | Location |
|------|----------|
| `onnxruntime_morphizen_ep.dll` | `%LOCAL%\bin\` — the ORT Execution Provider plugin |
| `morphizen_config.json` | `%LOCAL%\bin\` — EP pass pipeline config |
| Various morphizen pass DLLs | `%LOCAL%\bin\` |

### Config Files

Two pipeline configurations exist in `onnx-hipdnn-ep\etc\`:

| File | Pipeline | Use case |
|------|----------|----------|
| `morphizen_config_mlir.json` | MLIR (init → mlir-pass) | hip-compiler path |
| `morphizen_config.json` | ROCm (init → fuse_ROCm) | ROCm operator-level path |

The **installed** `%LOCAL%\bin\morphizen_config.json` currently uses the **MLIR pipeline**.

---

## Step 3: hip-compiler

- **Source**: `%WORKSPACE%\hip-compiler`
- **Build dir**: `C:\b\hip`

### Configure (fresh)

```bat
cmake -G Ninja ^
  -S %WORKSPACE%\hip-compiler ^
  -B C:\b\hip ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_PREFIX_PATH=%LOCAL% ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL% ^
  --fresh
```

### Build

```bat
cmake --build C:\b\hip --config Release --parallel
```

### Install

```bat
cmake --install C:\b\hip --config Release
```

### What gets installed

| File | Location |
|------|----------|
| `hip-compiler.dll` | `%LOCAL%\bin\` — compiler plugin loaded by MorphiZen EP |
| `hip-compile.exe` | `%LOCAL%\bin\` — standalone ONNX/MLIR → DLL compiler |
| `hip-opt.exe` | `%LOCAL%\bin\` — standalone MLIR pass runner |
| `test-model-dll.exe` | `%LOCAL%\bin\` — run compiled DLLs with random inputs |

### LIT Tests

```bat
ctest --test-dir C:\b\hip --config Release --verbose
```

---

## Step 4: test_onnx_runner

- **Source**: `%WORKSPACE%\test_onnx_runner`
- **Build dir**: `%WORKSPACE%\build\test_onnx_runner`

### Configure

```bat
cmake -G Ninja ^
  -S %WORKSPACE%\test_onnx_runner ^
  -B %WORKSPACE%\build\test_onnx_runner ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" ^
  -DCMAKE_PREFIX_PATH=%LOCAL% ^
  -DCMAKE_INSTALL_PREFIX=%LOCAL%
```

### Build & Install

```bat
cmake --build %WORKSPACE%\build\test_onnx_runner --config Release --parallel
cmake --install %WORKSPACE%\build\test_onnx_runner --config Release
```

Installs `test_onnx_runner.exe` to `%LOCAL%\bin\`.

---

## Running Tests

### A. hip-compiler LIT Tests

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
ctest --test-dir C:\b\hip --config Release --verbose
```

### B. hip-compiler Standalone Compile + Run

Compile an ONNX model or MLIR file to DLL:

```bat
%LOCAL%\bin\hip-compile.exe ^
  <input.onnx or input.mlir> ^
  --mode dll ^
  -o <output.dll> ^
  -v
```

Run the compiled DLL with random inputs:

```bat
%LOCAL%\bin\test-model-dll.exe <output.dll>
```

Example with the single-layer Llama model:

```bat
hip-compile.exe ^
  D:\Users\shoucair\shared_files\Llama-3.1-8B-awq-g128-int4-onnx-directml\single_op\mlir\simgle_layer.mlir ^
  --mode dll ^
  -o C:\b\single_layer_real_weights.dll ^
  -v

test-model-dll.exe C:\b\single_layer_real_weights.dll
```

### C. test_onnx_runner E2E with MorphiZen EP

#### Required Environment Variables

```bat
set USE_ORT_API_2_0=1
set MORPHIZEN_VITISAI_EP=onnxruntime_morphizen_ep.dll
set MORPHIZEN_EP_JSON_CONFIG=%LOCAL%\bin\morphizen_config.json
set PATH=%THEROCK%\bin;%PATH%
```

| Variable | Purpose |
|----------|---------|
| `USE_ORT_API_2_0=1` | Use `AppendExecutionProvider_V2` (new ORT API) instead of legacy VitisAI API |
| `MORPHIZEN_VITISAI_EP` | Filename of the EP DLL to load |
| `MORPHIZEN_EP_JSON_CONFIG` | Full path to the pass pipeline config JSON |
| `PATH` addition | TheRock ROCm bin dir (for HIP/MIOpen runtime DLLs) |

#### Run Command

```bat
cd /d %LOCAL%\bin

test_onnx_runner.exe ^
  D:\Users\shoucair\shared_files\Llama-3.1-8B-awq-g128-int4-onnx-directml\single_op\single_layer_v2_fix_decode_clean.onnx
```

#### Full One-Liner

```bat
cmd /c "cd /d %LOCAL%\bin && set PATH=%THEROCK%\bin;%PATH% && set USE_ORT_API_2_0=1 && set MORPHIZEN_VITISAI_EP=onnxruntime_morphizen_ep.dll && set MORPHIZEN_EP_JSON_CONFIG=%LOCAL%\bin\morphizen_config.json && test_onnx_runner.exe D:\Users\shoucair\shared_files\Llama-3.1-8B-awq-g128-int4-onnx-directml\single_op\single_layer_v2_fix_decode_clean.onnx 2>&1"
```

#### Expected Output (success)

```
Using ORT API 2.0 create session with MorphiZen EP...
Successfully added MorphiZenExecutionProvider EP
MemoryPoolingPass.cpp loaded into binary!
[CompilerDriver::compile] Input size: 1520456169 bytes
[ONNX→HIP] Discovered 14 constants...
[MemoryPooling] Pool size: 225280 bytes...
[MOCK] wrap_gather(...) / wrap_matmul(...) / wrap_group_query_attention(...) ...
done
```

Session create ~90s (MLIR compilation of 1.5 GB model), session run ~5ms.

---

## Wipe & Rebuild Cheat Sheet

### Wipe everything and rebuild all

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Wipe ORT
rmdir /s /q %WORKSPACE%\build\onnxruntime

:: Wipe onnx-hipdnn-ep
rmdir /s /q %WORKSPACE%\build

:: Wipe hip-compiler
rmdir /s /q C:\b\hip

:: Wipe test_onnx_runner
rmdir /s /q %WORKSPACE%\build\test_onnx_runner
```

Then re-run configure + build + install for each project in order (Steps 1 → 2 → 3 → 4).

### Incremental rebuild (no wipe)

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Rebuild + install ORT (only if ORT source changed)
:: Use Git Bash: ./build.bat ... (see Step 1)

:: Rebuild + install onnx-hipdnn-ep
cmake --build %WORKSPACE%\build --config Release --parallel
cmake --install %WORKSPACE%\build --config Release

:: Rebuild + install hip-compiler
cmake --build C:\b\hip --config Release --parallel
cmake --install C:\b\hip --config Release

:: Rebuild + install test_onnx_runner
cmake --build %WORKSPACE%\build\test_onnx_runner --config Release --parallel
cmake --install %WORKSPACE%\build\test_onnx_runner --config Release
```

---

## Test Model Files

| Model | Path |
|-------|------|
| Single-layer Llama decode (ONNX) | `D:\Users\shoucair\shared_files\Llama-3.1-8B-awq-g128-int4-onnx-directml\single_op\single_layer_v2_fix_decode_clean.onnx` |
| Single-layer Llama decode (MLIR bytecode) | `D:\Users\shoucair\shared_files\Llama-3.1-8B-awq-g128-int4-onnx-directml\single_op\mlir\simgle_layer.mlir` |

Model signature (5 inputs → 4 outputs):

```
Inputs:
  input_ids            : tensor<1x1xi64>
  attention_mask       : tensor<1x128xi64>
  position_ids         : tensor<1x1xi64>
  past_key_values.0.key   : tensor<1x8x127x128xf16>
  past_key_values.0.value : tensor<1x8x127x128xf16>

Outputs:
  present.0.key        : tensor<1x8x128x128xf16>
  present.0.value      : tensor<1x8x128x128xf16>
  layernorm_output     : tensor<1x1x4096xf16>
  mlp_output           : tensor<1x1x4096xf16>
```

---

## Known Issues

1. **Heap corruption on exit** — `test_onnx_runner` may exit with code `0xC0000374` after printing "done". The inference completes successfully; this is a cleanup issue in the mock runtime.

2. **`hip-opt.exe` file lock** — If `hip-opt.exe` is running in another terminal, `ninja` cannot overwrite it during rebuild. Kill the process first: `taskkill /F /IM hip-opt.exe`.

3. **Binary MLIR files on Windows** — The 1.5 GB `.mlir` file is MLIR bytecode (binary). File reading code must use `std::ios::binary` mode to avoid truncation at `0x1A` bytes.
