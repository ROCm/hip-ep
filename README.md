# onnx-hipdnn-ep

Unified ROCm Execution Provider for ONNX Runtime - GPU-accelerated operations using AMD's MIOpen and hipBLASLt libraries.

## Overview

This project implements a custom execution provider for ONNX Runtime that offloads operations to AMD GPUs using ROCm libraries. It combines the functionality of:
- **MIOpen**: Optimized convolution operations
- **hipBLASLt**: High-performance GEMM (matrix multiplication)

Both libraries share a single HIP stream for implicit operation fusion.

## Features

- **Unified EP Architecture**: Single execution provider supporting multiple operation types
- **Level-1/Level-2 Pass System**: Modular pattern matching with orchestration
- **Shared HIP Context**: Operations share the same GPU stream for implicit fusion
- **GPU Timeout Protection**: Prevents indefinite hangs with configurable timeouts
- **MIOpen Convolution**: Forward convolution with optional bias
- **hipBLASLt GEMM**: Matrix multiplication with epilogue support
- **Custom HIP Kernels**: Softmax, Tile, Transpose, Mul, Reshape operations

## Supported Operations

| Operation | Library | Status |
|-----------|---------|--------|
| Conv (2D) | MIOpen | ✅ Implemented |
| Conv + Bias | MIOpen | ✅ Implemented |
| Gemm | hipBLASLt | ✅ Implemented |
| Gemm + Bias | hipBLASLt | ✅ Implemented |
| MatMul | hipBLASLt | ✅ Implemented |
| Softmax | HIP Kernel | ✅ Implemented |
| Tile | HIP Kernel | ✅ Implemented |
| Transpose | HIP Kernel | ✅ Implemented |
| Mul | HIP Kernel | ✅ Implemented |
| Reshape | HIP Kernel | ✅ Implemented |

## Prerequisites

- Windows 10/11 or Linux
- AMD GPU with ROCm support (e.g., gfx1100, gfx1103, gfx1151)
- TheRock ROCm SDK (Windows) or ROCm (Linux)
- Visual Studio 2022 (Windows)
- CMake 3.29+
- Ninja build system
- Python 3.x
- Git (for fetching dependencies)

## Quick Start

### Windows

1. **Install TheRock ROCm SDK:**
   ```batch
   REM Download from https://therock-nightly-tarball.s3.amazonaws.com/index.html
   REM Choose the version matching your GPU architecture (e.g., gfx1151)
   REM Extract to D:\Develop\m\dist\therock (or C:\Develop\m\dist\therock)
   set THEROCK_DIST=D:\Develop\m\dist\therock
   ```

2. **Build the project (default architecture):**
   ```batch
   cd onnx-hipdnn-ep
   build.bat
   ```

3. **Build with specific GPU architecture:**
   
   For AMD GPUs, you must specify the correct architecture. Common architectures:
   - `gfx1100` - Radeon RX 7900 series
   - `gfx1103` - Radeon 780M/760M (iGPU)
   - `gfx1151` - Radeon PRO W7000 series
   
   ```batch
   REM Method 1: Set environment variable before build
   set HIP_ARCHITECTURES=gfx1151
   build.bat
   
   REM Method 2: Direct CMake configuration
   cmake -DHIP_ARCHITECTURES=gfx1151 -B build -S .
   cmake --build build
   ```

4. **Configure GPU timeout (optional):**
   ```batch
   REM Set timeout to 10 seconds (default is 5 seconds)
   set MORPHIZEN_GPU_TIMEOUT_MS=10000
   ```

### Linux

```bash
# Set up ROCm environment
export THEROCK_DIST=/opt/rocm

# Build with specific GPU architecture
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DHIP_ARCHITECTURES=gfx1100 ..
cmake --build .
```

## GPU Architecture Selection

**Important:** The HIP kernels must be compiled for your specific GPU architecture. Using the wrong architecture will result in runtime crashes (access violation errors).

To find your GPU architecture:
```batch
REM On Windows with TheRock
%THEROCK_DIST%\bin\rocminfo.exe | findstr "gfx"

REM On Linux
rocminfo | grep gfx
```

Set the architecture during CMake configuration:
```batch
cmake -DHIP_ARCHITECTURES=gfx1151 <build_dir>
```

For incremental builds after changing architecture:
```batch
REM Reconfigure with new architecture
cmake -DHIP_ARCHITECTURES=gfx1151 <build_dir>

REM Rebuild kernels and relink
cmake --build <build_dir>
```

## Project Structure

```
onnx-hipdnn-ep/
├── cmake/                          # CMake configuration
├── level-1-pass-rocm/             # Level-1 pass (orchestrator)
├── level-2-pass-rocm-conv/        # Level-2 Conv pass (MIOpen)
├── level-2-pass-rocm-gemm/        # Level-2 Gemm pass (hipBLASLt)
├── custom-op-rocm/                # Runtime custom op
├── proto/                         # Protobuf definitions
├── patterns/                      # ONNX patterns
├── test/                          # Tests
├── doc/                           # Documentation
├── etc/                           # Configuration
└── tools/                         # Build tools
```

## Architecture

```
                    ┌─────────────────────────────┐
                    │    Level-1 Pass (ROCm)      │
                    │  • GPU availability check   │
                    │  • Orchestrates sub-passes  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │                             │
              ┌─────▼─────┐               ┌───────▼─────┐
              │ Level-2   │               │ Level-2     │
              │ Conv Pass │               │ Gemm Pass   │
              │ (MIOpen)  │               │ (hipBLASLt) │
              └─────┬─────┘               └──────┬──────┘
                    │                            │
                    └────────────┬───────────────┘
                                 ▼
                    ┌─────────────────────────────┐
                    │     Custom Op (ROCm)        │
                    │  • Shared HIP stream        │
                    │  • ConvExecutor             │
                    │  • GemmExecutor             │
                    └─────────────────────────────┘
```

## Testing

After building, you can verify the installation with the included test tools.

### Run GQA Test

```batch
REM Set environment
set THEROCK_DIST=D:\Develop\m\dist\therock
set PATH=%THEROCK_DIST%\bin;%PATH%
set MORPHIZEN_EP_DLL=D:\Develop\m\local\bin\onnxruntime_morphizen_ep.dll

REM Run test
test_gqa.exe <model.onnx>

REM With options
test_gqa.exe -b 1 -s 128 -i 10 -v <model.onnx>
```

### Model Verification (CPU vs GPU comparison)

```batch
REM Verify model outputs match between CPU and GPU
model_verifier.exe <model.onnx>

REM With custom tolerance
model_verifier.exe -t 0.01 <model.onnx>
```

### Test Options

| Option | Description |
|--------|-------------|
| `-b <size>` | Batch size (default: 1) |
| `-s <len>` | Sequence length (default: 128) |
| `-i <num>` | Number of iterations (default: 10) |
| `-w <num>` | Warmup iterations (default: 3) |
| `-t <val>` | Tolerance for verification (default: 0.1) |
| `-v` | Verbose output |
| `-n` | Disable EP (CPU only) |

## Troubleshooting

### Access Violation (0xC0000005) in hipLaunchKernel

This error typically means the HIP kernels were compiled for a different GPU architecture than your hardware.

**Solution:** Rebuild with the correct `HIP_ARCHITECTURES`:
```batch
cmake -DHIP_ARCHITECTURES=<your_arch> <build_dir>
cmake --build <build_dir>
```

### TheRock ROCm SDK not found

The build script checks these locations:
- `D:\Develop\m\dist\therock`
- `C:\Develop\m\dist\therock`
- `C:\dist\therock`

Set `THEROCK_DIST` environment variable if your SDK is elsewhere.

### GitHub Authentication Error (SAML SSO)

If you see "The 'ROCm' organization has enabled or enforced SAML SSO", re-authorize Git Credential Manager:
```batch
git credential-manager github login
```

## Documentation

- [01_DESIGN.md](doc/01_DESIGN.md) - Architecture and design details
- [02_LEVEL1_PASS_DESIGN.md](doc/02_LEVEL1_PASS_DESIGN.md) - Level-1 pass implementation
- [03_GROUPING_ALGORITHM.md](doc/03_GROUPING_ALGORITHM.md) - Pattern grouping algorithm
- [04_TEST_STATUS_REPORT.md](doc/04_TEST_STATUS_REPORT.md) - Test status and results
- [05_GPU_TIMEOUT_HANDLING.md](doc/05_GPU_TIMEOUT_HANDLING.md) - GPU timeout protection
- [10_BUILD.md](doc/10_BUILD.md) - Detailed build instructions

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
