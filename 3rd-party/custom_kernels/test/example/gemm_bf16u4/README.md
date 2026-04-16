# MatMulNBits Test Suite

Tests for `hip_matmul_nbits()` — fused GEMM + uint4 dequantization kernel.

```
C[M,N] = A[M,K] × dequant(B_packed[N,K/2])^T
```

## Prerequisites

- **HIP SDK** (ROCm for Windows or Linux)
- **Python 3** with NumPy (`pip install numpy`)
- **GNU Make**

Edit `HIP_SDK` and `OFFLOAD` in the Makefile if your paths or GPU arch differ:

```makefile
OFFLOAD := --offload-arch=gfx1151          # change to your GPU arch
HIP_SDK ?= C:\Users\...\ROCm\7.1          # change to your HIP SDK path
```

## Quick Start

```bash
# Single shape test (generates data + builds + runs)
make test SIZE=128x2880x5120 NO_ZEROS=1

# Model sweep (all shapes from a model config)
make test_model MODEL_JSON=test_model/gpt_oss_20b.json NO_ZEROS=1
```

## Test Modes

### 1. Single Shape (`make test`)

Generate random test data, build the kernel, run benchmark + verification.

```bash
make test SIZE=<M>x<K>x<N> [GS=128] [NO_ZEROS=1]
```

| Variable | Default | Description |
|---|---|---|
| `SIZE` | `128x128x128` | Matrix dimensions M×K×N |
| `GS` | `128` | Quantization group size |
| `NO_ZEROS` | *(empty)* | Set `1` to disable zero points |

Examples:

```bash
make test SIZE=1x2880x201088 NO_ZEROS=1       # decode (GEMV path, M=1)
make test SIZE=128x2880x5120 NO_ZEROS=1       # prefill 128 (WMMA path)
make test SIZE=1024x2880x5120 NO_ZEROS=1      # prefill 1024 (WMMA path)
```

### 2. Model Sweep (`make test_model`)

Iterate through all shape combinations defined in a model config JSON.
A global warm-up runs once before benchmarking. Each shape reports performance
metrics and pass/fail verification status.

```bash
make test_model MODEL_JSON=<path> [GS=128] [NO_ZEROS=1]
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_JSON` | `test_model/gpt_oss_20b.json` | Model config file |
| `MODEL_DATA_DIR` | `data_model` | Data output directory |

This runs two steps:
1. `gen_model_data.py` generates test data for every (M, K, N) combination
2. The test binary loads each shape's data, benchmarks, and verifies

### 3. True-Data Mode (`make test_real`)

Test with real model weights exported from a framework (e.g., ONNX Runtime).

```bash
make test_real DATA_PATH=true_data/layer_0/o_proj
```

Requires a `shape.json` and corresponding binary files in the data folder.

### 4. Preset Suite (`make test_all`)

Runs a set of small-to-medium shapes for quick sanity checks:

```bash
make test_all
```

### 5. Large-N Benchmark (`make bench_large_n`)

Benchmarks M=512..3072 at K=2880, N=201088 (no zeros):

```bash
make bench_large_n
```

## Supported Model Configs

Model configs are JSON files in `test_model/`. Each defines a set of shapes to test.

### GPT-OSS-20B (`test_model/gpt_oss_20b.json`)

```json
{
    "M_array": [1, 128, 256, 512, 1024, 2048, 3072],
    "KN_pairs": {
        "K": [2880, 4096, 2880, 2880],
        "N": [201088, 2880, 5120, 32]
    }
}
```

- **M_array**: batch sizes / sequence lengths to test (7 values)
- **KN_pairs**: weight matrix dimensions, K and N are paired (4 pairs)
- **Total shapes**: 7 × 4 = 28 combinations

The kernel automatically selects the execution path based on M:

| M | Path | Scenario |
|---|---|---|
| 1 | GEMV (K-parallel reduction) | Decode, single token |
| 128+ | WMMA (matrix multiply hardware) | Prefill |

### Adding a New Model

Create a JSON file with the same structure:

```json
{
    "M_array": [1, 64, 128, 256],
    "KN_pairs": {
        "K": [4096, 4096],
        "N": [4096, 11008]
    }
}
```

Then run:

```bash
make test_model MODEL_JSON=test_model/my_model.json GS=128 NO_ZEROS=1
```

## Build Targets

| Target | Description |
|---|---|
| `make direct` | Build only (direct mode, no library needed) |
| `make lib` | Build the kernel as a static library |
| `make test` | Generate data + build + run (direct mode) |
| `make test_lib` | Same as `test` but using library mode |
| `make test_model` | Model sweep (all shapes from JSON) |
| `make test_real` | Test with real exported weights |
| `make test_all` | Preset sanity-check suite |
| `make bench_large_n` | Large-N benchmarks |
| `make gendata` | Generate test data only (no build/run) |
| `make gendata_model` | Generate model sweep data only |
| `make asm` | Generate ISA assembly for inspection |
| `make clean` | Remove build artifacts and data |

## Output Format

Each test prints:

```
=== Test MatMulNBits (WMMA) M=128 N=5120 K=2880 group_size=128 (no zeros) ===
  Loaded input data from data/
  Pre-warmup (2000 iters)... done (46 ms, 0.023 ms/iter)
  Warmup... OK (0.31 ms), iters=200
  Benchmarking (5 rounds x 200 iters)... done

  === Performance ===
  Median: 0.072 ms, 29441 GFLOPS, 207.5 GB/s

  === GPU vs Python Reference ===
  Verified 655360 elements, 0 errors
  Result: PASSED
```

## Directory Structure

```
gemm_bf16u4/
├── Makefile                      # Build and test automation
├── README.md                     # This file
├── test_matmul_nbits.cpp         # C++ test driver
├── gen_matmul_nbits_data.py      # Single-shape data generator
├── gen_model_data.py             # Model sweep data generator
├── verify_true_data.py           # True-data verification utility
└── test_model/
    └── gpt_oss_20b.json          # GPT-OSS-20B model config
```
