<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX HIP DNN Execution Provider

An implementation of HIP DNN operations in the MorphiZen framework.

This project demonstrates the integration of HIP (Heterogeneous-compute Interface for Portability) DNN operations within the MorphiZen optimization framework for ONNX Runtime.

---

## Features

- **Unified EP Architecture**: Single execution provider supporting multiple operation types
- **Level-1/Level-2 Pass System**: Modular pattern matching with orchestration
- **Shared HIP Context**: Operations share the same GPU stream for implicit fusion
- **GPU Timeout Protection**: Prevents indefinite hangs with configurable timeouts
- **MIOpen Convolution**: Forward convolution with optional bias
- **hipBLASLt GEMM**: Matrix multiplication with epilogue support
- **Custom HIP Kernels**: Softmax, Tile, Transpose, Mul, Reshape operations

---

## Supported Operations

| Operation | Library | Status |
|-----------|---------|--------|
| Conv (2D) | MIOpen | Implemented |
| Conv + Bias | MIOpen | Implemented |
| Gemm | hipBLASLt | Implemented |
| Gemm + Bias | hipBLASLt | Implemented |
| MatMul | rocBLAS | Implemented |
| Mul | HIPRTC | Implemented |
| Softmax | HIPRTC | Implemented |
| Reshape | HIP Memory | Implemented |
| Transpose | HIPRTC | Implemented |
| Tile | HIPRTC | Implemented |

---

## Project Design

### Components

- **level-1-pass-rocm**: Level-1 orchestrator pass for ROCm operations
- **level-2-pass-rocm-conv**: Level-2 Conv pass (MIOpen)
- **level-2-pass-rocm-gemm**: Level-2 Gemm pass (hipBLASLt)
- **level-2-pass-rocm-matmul**: Level-2 MatMul pass (rocBLAS)
- **level-2-pass-rocm-mul**: Level-2 Mul pass (HIPRTC)
- **level-2-pass-rocm-softmax**: Level-2 Softmax pass (HIPRTC)
- **level-2-pass-rocm-reshape**: Level-2 Reshape pass
- **level-2-pass-rocm-transpose**: Level-2 Transpose pass (HIPRTC)
- **level-2-pass-rocm-tile**: Level-2 Tile pass (HIPRTC)
- **custom-op-rocm**: Custom operator implementations using HIP
- **kernels**: GPU kernel implementations (HIPRTC)
- **proto**: Protocol buffer definitions
- **test**: Test suite for validation

### Architecture

```
                    ┌─────────────────────────────┐
                    │    Level-1 Pass (ROCm)      │
                    │  • GPU availability check   │
                    │  • Orchestrates sub-passes  │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │           │           │           │           │
    ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
    │ Level-2   │ │Level-2│ │ Level-2   │ │Level-2│ │ Level-2   │
    │ Conv Pass │ │ Gemm  │ │ MatMul    │ │ Mul   │ │ Softmax   │
    │ (MIOpen)  │ │(hBLT) │ │ (rocBLAS) │ │(HIPRTC│ │ (HIPRTC)  │
    └─────┬─────┘ └───┬───┘ └─────┬─────┘ └───┬───┘ └─────┬─────┘
          │           │           │           │           │
          └───────────┴───────────┼───────────┴───────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │     Custom Op (ROCm)        │
                    │  • Shared HIP stream        │
                    │  • ConvExecutor             │
                    │  • GemmExecutor             │
                    │  • HIPRTC Kernels           │
                    └─────────────────────────────┘
```

---

## Prerequisites

### System Requirements
- Windows 10/11 with AMD GPU (ROCm support)
- Visual Studio 2022 with C++ workload
- CMake 3.29+
- Git with Git Bash
- Python 3.8+

---

## Directory Structure

After completing the build instructions below, your workspace will have the following structure:

```
workspace/
├── therock/                  # TheRock ROCm SDK (extracted from tarball)
│   ├── bin/                  # Runtime DLLs (MIOpen.dll, hiprtc.dll, etc.)
│   └── lib/llvm/bin/         # LLVM tools (amdgpu-arch.exe)
├── onnxruntime/              # ONNX Runtime source code (git clone)
├── build/
│   ├── onnxruntime/          # ONNX Runtime build artifacts
│   └── onnx-hipdnn-ep/       # onnx-hipdnn-ep build artifacts
├── local/                    # ONNX Runtime installation (CMAKE_PREFIX_PATH)
│   ├── bin/                  # onnxruntime.dll, onnxruntime_morphizen_ep.dll, test_gqa.exe
│   └── lib/cmake/            # CMake configuration files
└── onnx-hipdnn-ep/           # This project (git clone)
    ├── 3rd-party/morphizen/  # MorphiZen framework (git submodule)
    ├── test/models/     # Test models (gqa_layer_00.onnx)
    └── etc/                  # Configuration files (morphizen_config.json)
```

---

## Build Instructions

### Step 1: Setup TheRock ROCm SDK

TheRock SDK provides HIP/ROCm runtime for Windows.

**Download:** https://therock-nightly-tarball.s3.amazonaws.com/index.html

1. **Determine your GPU architecture** (before downloading):

   Open **Device Manager** → **Display adapters** to find your AMD GPU model, then select the matching GFX series:

   | GPU Model | GFX Series | TheRock Tarball |
   |-----------|------------|-----------------|
   | Radeon RX 7900/7800/7700/7600 | gfx110X | `therock-dist-windows-gfx110X-all-*.tar.gz` |
   | Radeon RX 6900/6800/6700/6600 | gfx103X | `therock-dist-windows-gfx103X-all-*.tar.gz` |
   | Radeon 880M/780M (Strix Point) | gfx115X | `therock-dist-windows-gfx115X-all-*.tar.gz` |
   | Radeon 890M (Strix Halo) | gfx120X | `therock-dist-windows-gfx120X-all-*.tar.gz` |

2. **Create workspace and extract TheRock**:
   ```bash
   mkdir workspace
   cd workspace

   # Extract TheRock tarball to workspace/therock
   mkdir therock
   tar -xzf /path/to/therock-dist-windows-gfx115X-all-*.tar.gz -C therock
   ```

3. **Verify installation**:
   ```bash
   # Verify GPU architecture detection (IMPORTANT: note this value for build!)
   ./therock/lib/llvm/bin/amdgpu-arch.exe
   # Example output: gfx1151

   # Alternative: hipInfo shows architecture as 'gcnArchName'
   ./therock/bin/hipInfo.exe | grep gcnArchName
   # Example output: gcnArchName: gfx1151

   # Verify HIP configuration (version, paths, compiler)
   ./therock/bin/hipconfig.exe --full
   ```

### Step 2: Build ONNXRuntime

To build ONNX Runtime, follow the [official documentation](https://onnxruntime.ai/docs/build/inferencing.html).

It is recommended to use Git Bash. The commands below have only been tested in Git Bash.

#### Download onnxruntime

```bash
cd workspace
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNX Runtime

```bash
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local --disable_memleak_checker
cmake --build ../build/onnxruntime/Release/ --target install
```

### Step 3: Build onnx-hipdnn-ep

#### Download onnx-hipdnn-ep

```bash
cd workspace
git clone --recursive https://github.com/ROCm/onnx-hipdnn-ep.git
```

> **Note**: The `--recursive` flag is required to clone the `3rd-party/morphizen` submodule. If you already cloned without it, run:
> ```bash
> cd onnx-hipdnn-ep
> git submodule update --init --recursive
> ```

#### Known Issue

**nlohmann_json Package Not Found**

```
Error Message:
CMake Error: Could not find a package configuration file provided by "nlohmann_json"

Root Cause:
TheRock SDK's nlohmann_json CMake configuration file contains problematic INTERFACE_SOURCES attribute.

Solution:
File Path: $PWD/../therock/share/cmake/nlohmann_json/nlohmann_jsonTargets.cmake
Modifications:
Open file and find set_target_properties(nlohmann_json::nlohmann_json ...)
Remove the INTERFACE_SOURCES line (if it exists)
```

#### Detect HIP Architecture (CRITICAL)

> **⚠️ IMPORTANT FOR DEVELOPERS AND AI TOOLS:**
> You **MUST** set `HIP_ARCHITECTURES` to match your GPU. Failure to do so will result in
> runtime errors like `Exception Code: 0xC0000005` (Access Violation) in `hipLaunchKernel`.
> The HIP kernels are compiled for a specific GPU architecture; mismatched architectures
> cause kernel launch failures at runtime.

**Detect your GPU architecture using the SDK:**

```bash
# Using amdgpu-arch from TheRock SDK (recommended - outputs just the arch)
$THEROCK_DIST/lib/llvm/bin/amdgpu-arch
# Example output: gfx1151

# Alternative: Using hipInfo (look for gcnArchName)
$THEROCK_DIST/bin/hipInfo | grep gcnArchName
# Example output: gcnArchName: gfx1151
```

**Common HIP architectures:**

| GPU Model | Architecture |
|-----------|--------------|
| Radeon RX 7900 XTX/XT | gfx1100 |
| Radeon RX 7800/7700 | gfx1101, gfx1102 |
| Radeon RX 7600 | gfx1102 |
| Radeon RX 6900/6800 | gfx1030 |
| Radeon RX 6700/6600 | gfx1031, gfx1032 |
| Radeon 890M (Strix Halo) | gfx1201 |
| Radeon 880M/780M (Strix Point) | gfx1150, gfx1151 |

#### Configure and build

**Using Bash (Git Bash on Windows):**

```bash
cd onnx-hipdnn-ep
export THEROCK_DIST=$PWD/../therock

# CRITICAL: Detect and set HIP architecture
export HIP_ARCH=$($THEROCK_DIST/lib/llvm/bin/amdgpu-arch)
echo "Detected HIP architecture: $HIP_ARCH"

cmake \
  -B ../build/onnx-hipdnn-ep -S . \
  -DTHEROCK_DIST=$THEROCK_DIST \
  -DCMAKE_PREFIX_PATH=$PWD/../local \
  -DCMAKE_INSTALL_PREFIX=$PWD/../local \
  -DHIP_PLATFORM=amd \
  -DHIP_ARCHITECTURES=$HIP_ARCH

# Build Release version (recommended)
cmake --build ../build/onnx-hipdnn-ep --config Release --target install --parallel
```

**Using PowerShell:**

```powershell
cd onnx-hipdnn-ep
$env:THEROCK_DIST = "$PWD\..\therock"
$env:HIP_PLATFORM = "amd"

# CRITICAL: Detect and set HIP architecture
$HIP_ARCH = & "$env:THEROCK_DIST\lib\llvm\bin\amdgpu-arch.exe"
Write-Host "Detected HIP architecture: $HIP_ARCH"

cmake -B ..\build\onnx-hipdnn-ep -S . `
  -DTHEROCK_DIST="$env:THEROCK_DIST" `
  -DCMAKE_PREFIX_PATH="$PWD\..\local" `
  -DCMAKE_INSTALL_PREFIX="$PWD\..\local" `
  -DHIP_PLATFORM=amd `
  -DHIP_ARCHITECTURES=$HIP_ARCH

# Build Release version (recommended)
cmake --build ..\build\onnx-hipdnn-ep --config Release --target install --parallel
```

#### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `THEROCK_DIST` | (required) | Path to TheRock ROCm SDK installation |
| `CMAKE_PREFIX_PATH` | - | Path to ONNX Runtime installation (for find_package) |
| `CMAKE_INSTALL_PREFIX` | - | Installation directory for built artifacts |
| `HIP_PLATFORM` | `amd` | HIP platform (use `amd` for AMD GPUs) |
| `HIP_ARCHITECTURES` | **(required)** | GPU architecture (e.g., `gfx1151`). **MUST match your GPU!** Detect using `$THEROCK_DIST/lib/llvm/bin/amdgpu-arch`. Mismatched architectures cause runtime crashes. |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `THEROCK_DIST` | Path to TheRock SDK installation |
| `HIP_PLATFORM` | Set to `amd` for AMD GPU support |
| `MORPHIZEN_DEBUG_ROCM` | Debug level (1=basic, 2=verbose) |
| `MORPHIZEN_ROCM_EN_LVL1_MERGE` | Enable Level-1 subgraph merging (1=enabled) |
| `MORPHIZEN_DRY_RUN` | Dry run mode without GPU execution (1=enabled) |
| `MORPHIZEN_EP_JSON_CONFIG` | Path to custom morphizen_config.json |

---

## Testing

### Run Sample Test show call MIOpen/hipBLASLt/HIP custom kernel/...

```bash
# Set environment
export PATH="$THEROCK_DIST/bin:$PATH"

# Run test
cd ../build/onnx-hipdnn-ep/bin/Release/
python ../../../../onnx-hipdnn-ep/test/gen_sample_model.py
./ort_integration_test.exe
```

### Run GQA Test

```bash
# Set environment
export PATH="$THEROCK_DIST/bin:$PATH"

# Run test
cd /path/to/onnx-hipdnn-ep
cd ../local/bin
./test_gqa.exe ../../onnx-hipdnn-ep/test/models/gqa_layer_00.onnx
```

```powershell
# Set environment
$env:PATH = "$env:THEROCK_DIST\bin;$env:PATH"

# Run test
cd ..\local\bin
.\test_gqa.exe ..\..\onnx-hipdnn-ep\test\models\gqa_layer_00.onnx
```

### Benchmark Results

```
=== Benchmark Results ===
  Iterations: 1
  Mean latency: 53.50 ms
  Std dev: 0.00 ms
  Min latency: 53.50 ms
  Max latency: 53.50 ms
  Median: 53.50 ms
  P90: 53.50 ms
  P99: 53.50 ms
  Throughput: 18.69 infer/sec
```

### Run ORT Integration Unit Test

Tests MIOpen Conv, hipBLASLt Gemm, and basic operations:

```bash
# Set environment
export PATH="$THEROCK_DIST/bin:$PATH"
export ENABLE_CACHE_CONTEXT=1  # optional, 1:generate EP Context model

# Generate sample model and run test
cd ../local/bin
python ../../onnx-hipdnn-ep/test/gen_sample_model.py
./ort_integration_test.exe
```

```powershell
# Set environment
$env:PATH = "$env:THEROCK_DIST\bin;$env:PATH"
$env:ENABLE_CACHE_CONTEXT = "1"  # optional, 1:generate EP Context model

# Generate sample model and run test
cd ..\local\bin
python ..\..\onnx-hipdnn-ep\test\gen_sample_model.py
.\ort_integration_test.exe
```

**Expected Output:**
```
[==========] Running 2 tests from 1 test suite.
[ RUN      ] OrtIntegrationTest.LoadMorphiZenProvider
[       OK ] OrtIntegrationTest.LoadMorphiZenProvider (234 ms)
[ RUN      ] OrtIntegrationTest.MorphiZenProviderInference
[Test] Max difference between CPU and GPU: 2.98023e-07
[       OK ] OrtIntegrationTest.MorphiZenProviderInference (343 ms)
[  PASSED  ] 2 tests.
```

If you set `ENABLE_CACHE_CONTEXT=1`, an EP context model (`sample_ctx.onnx`) will be generated in the same directory as `sample.onnx`. This cached model contains pre-compiled graph optimizations and can be loaded directly for faster startup in subsequent runs.

> **Note:** You must delete `sample_ctx.onnx` before re-running the test if you want to regenerate it.

---

## Troubleshooting

### Runtime Crash: Access Violation in hipLaunchKernel

**Symptoms:**
```
Exception Code: 0xC0000005
hipLaunchKernel() + 0x871F0 byte(s)
```

**Cause:** HIP kernels were compiled for a different GPU architecture than the one you're running on.

**Solution:**
1. Detect your GPU architecture:
   ```bash
   $THEROCK_DIST/lib/llvm/bin/amdgpu-arch
   ```
2. Rebuild with the correct architecture:
   ```bash
   cmake -B ../build/onnx-hipdnn-ep -DHIP_ARCHITECTURES=<your-arch>
   cmake --build ../build/onnx-hipdnn-ep --config Release
   ```

> **Note for AI Tools:** When building this project, always detect the HIP architecture
> using `amdgpu-arch` from the TheRock SDK and pass it via `-DHIP_ARCHITECTURES=<arch>`.
> Never omit this option or use a hardcoded default.

### MLIR Configuration Error: MLIRTargets.cmake Not Found

**Symptoms:**
```
CMake Error: include could not find requested file: MLIRTargets.cmake
```

**Cause:** Corrupted CMake cache after partial build or reconfiguration.

**Solution:**
1. Clear the LLVM build cache:
   ```bash
   rm -rf ../build/onnx-hipdnn-ep/_deps/llvm-project-*
   ```
2. Reconfigure and rebuild.

---

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
