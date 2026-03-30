# MatMulNBits — Fused GEMM + Dequantization (RDNA3 WMMA)

## Overview

Tests the `hip_matmul_nbits()` API with the RDNA3 WMMA fast path.

Computes `C = A × dequant(B_packed)^T` using RDNA3 WMMA instructions.

| Tensor     | Type          | Layout                        |
|------------|---------------|-------------------------------|
| A          | FP16          | col-major (M × K), stride M   |
| B_packed   | UINT4 packed  | (N × K/2) bytes, row-major    |
| scales     | FP16          | (N × num_groups_k), row-major |
| zeros      | FP16 (optional)| (N × num_groups_k), row-major |
| C          | FP16          | col-major (M × N), stride M   |

Dequantization: `val = (uint4_val - zero) * scale` (zeros = NULL → zero=0)

WMMA fast path constraints: M % 128 == 0, N % 128 == 0, K % 32 == 0

## Prerequisites

- **HIP SDK / ROCm** (with `hipcc`)
- **Python 3** + **NumPy** (for data generation)
- **RDNA3+ GPU** (gfx11xx / gfx12xx)
- **GNU Make** (Windows: `choco install make`)

## Quick Start

One command to build + generate data + run verification:

```powershell
cd custom_kernels\example\matmul_nbits
make test                                   # default 128x128x128, gs=128
make test SIZE=256x512x256 GS=128           # custom size
make test SIZE=128x128x128 GS=128 NO_ZEROS=1  # no zero points
make test_all                               # all preset sizes
```

Step by step:

```powershell
make                    # build (direct mode)
make gendata            # generate test data
make run                # run (needs built binary + data)
make clean              # clean all artifacts
```

### Makefile Variables

| Variable | Default                    | Description              |
|----------|---------------------------|--------------------------|
| OFFLOAD  | `--offload-arch=gfx1150`  | GPU architecture         |
| HIP_SDK  | `C:/AMD/ROCm/7.1`        | HIP SDK path             |
| SIZE     | `128x128x128`             | Matrix dimensions MxKxN  |
| GS       | `128`                     | Quantization group size  |
| NO_ZEROS | (empty)                   | Set to `1` to disable ZP |

## Usage

```
test_matmul_nbits.exe [MxKxN] [group_size] [data_dir] [--no-zeros]
```

| Argument    | Default       | Description                           |
|-------------|---------------|---------------------------------------|
| MxKxN       | 128x128x128   | Matrix dimensions (M×K×N)             |
| group_size  | 128           | Quantization group size along K       |
| data_dir    | data          | Binary data file directory            |
| --no-zeros  | (not set)     | Disable zero points (zero = 0)        |

## API

```c
#include "hip_custom_kernels.h"

int hip_matmul_nbits(
    void* stream,           // hipStream_t
    const void* A,          // FP16 col-major (WMMA path)
    const void* B,          // uint4 packed
    const void* scales,     // FP16 per-group scales
    const void* zero_points,// FP16 per-group zeros, or NULL (WMMA path)
    const void* bias,       // FP16 bias [N], or NULL
    void* output,           // FP16 col-major output (WMMA path)
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,    // must be 1 for WMMA path
    int64_t bits,           // must be 4
    int64_t block_size,     // quantization group size
    int64_t element_size_bytes); // must be 2 (fp16)
```

Returns `0` on success, non-zero `hipError_t` on failure.

The WMMA fast path is automatically selected when:
- `batch_count == 1`
- `M % 128 == 0`, `N % 128 == 0`, `K % 32 == 0`
- `element_size_bytes == 2` (FP16)

Otherwise, a naive fallback kernel is used (row-major layout, uint8 zero points).

## Expected Output

```
custom_kernels MatMulNBits (WMMA Fused GEMM + Dequantization) Verification
==========================================================================
GPU: AMD Radeon(TM) 890M Graphics (arch: gfx1150)

=== Test MatMulNBits (WMMA) M=128 N=128 K=128 group_size=128 ===
  Loaded input data from data/
  Warmup... OK (1.10 ms), iters=200
  Benchmarking (5 rounds x 200 iters)... done

  === Performance ===
  Median: 0.006 ms, 676.774 GFLOPS, 11.979 GB/s

  === GPU vs Python Reference ===
  Verified 16384 elements, 0 errors
  Result: PASSED

==========================================================================
Overall: ALL PASSED
```
