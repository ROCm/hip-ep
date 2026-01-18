# morphizen-rocm

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

## Supported Operations

| Operation | Library | Status |
|-----------|---------|--------|
| Conv (2D) | MIOpen | ✅ Implemented |
| Conv + Bias | MIOpen | ✅ Implemented |
| Gemm | hipBLASLt | ✅ Implemented |
| Gemm + Bias | hipBLASLt | ✅ Implemented |

## Prerequisites

- Windows 10/11 or Linux
- AMD GPU with ROCm support
- TheRock ROCm SDK (Windows) or ROCm (Linux)
- Visual Studio 2022 (Windows)
- CMake 3.29+
- Ninja build system
- Python 3.x

## Quick Start

### Windows

1. Install TheRock ROCm SDK:
   ```batch
   REM Download from https://therock-nightly-tarball.s3.amazonaws.com/index.html
   REM Extract to C:\Develop\m\dist\therock
   set THEROCK_DIST=C:\Develop\m\dist\therock
   ```

2. Build the project:
   ```batch
   cd morphizen-rocm
   build.bat
   ```

3. Configure GPU timeout (optional):
   ```batch
   REM Set timeout to 10 seconds (default is 5 seconds)
   set MORPHIZEN_GPU_TIMEOUT_MS=10000
   ```

### Linux

```bash
# Set up ROCm environment
export THEROCK_DIST=/opt/rocm

# Build
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Project Structure

```
morphizen-rocm/
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

## Documentation

- [01_DESIGN.md](doc/01_DESIGN.md) - Architecture and design details
- [02_LEVEL1_PASS_DESIGN.md](doc/02_LEVEL1_PASS_DESIGN.md) - Level-1 pass implementation
- [03_GROUPING_ALGORITHM.md](doc/03_GROUPING_ALGORITHM.md) - Pattern grouping algorithm
- [04_TEST_STATUS_REPORT.md](doc/04_TEST_STATUS_REPORT.md) - Test status and results
- [05_GPU_TIMEOUT_HANDLING.md](doc/05_GPU_TIMEOUT_HANDLING.md) - GPU timeout protection

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
