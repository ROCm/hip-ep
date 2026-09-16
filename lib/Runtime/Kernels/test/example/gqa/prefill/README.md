# GQA Prefill Test Suite

Tests for the fused FA-2 WMMA prefill kernels (`sq > 1`), which the runtime
selects by head_dim: v5 at 64, v7 at 128, v8 at 256.

Both programs are standalone: they compile `gqa_kernel.hip` directly and call
the kernel entry, so nothing needs the EP or a model to be built first.

## Prerequisites

- **HIP SDK** (ROCm for Windows or Linux) with `hipcc` on `PATH`
- A supported AMD GPU; the commands below target `gfx1151`, change
  `--offload-arch` to match yours

All commands are run from this directory:

```bash
cd lib/Runtime/Kernels/test/example/gqa/prefill
```

`../../../..` is the `Kernels` root, which is where the kernel source and its
headers live.

## Quick start

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_prefill.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_prefill.exe
./test_gqa_prefill.exe
```

The last line is the verdict.

## Programs

| Program | What it does |
|---|---|
| `test_gqa_prefill.cpp` | fp16 correctness and TTFT over the prefill shapes, including sliding-window and head-sink variants |
| `test_gqa_prefill_i8.cpp` | The same against an INT8 KV cache |

### `test_gqa_prefill` (fp16)

Checks each case against an fp32 CPU reference and times it.

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_prefill.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_prefill.exe
./test_gqa_prefill.exe --iters 50
```

| Option | Default | Meaning |
|---|---|---|
| `--iters <n>` | see source | Timed iterations per case |

### `test_gqa_prefill_i8` (INT8 KV)

The INT8 KV cache is dequantized once into an fp16 scratch buffer and the fp16
prefill kernels run on that, so this exercises the runtime path rather than a
separate kernel.

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_prefill_i8.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_prefill_i8.exe
./test_gqa_prefill_i8.exe --all
```

| Option | Default | Meaning |
|---|---|---|
| `--all` | off | Run the full shape matrix instead of a single shape |
| `--iters <n>` | see source | Timed iterations per case |
| `--seed <n>` | 1234 | Input RNG seed |
| `--md <file>` | none | Write a Markdown result table to the given file |
| `--h/--g/--d/--sq` | — | Run one custom shape instead of the matrix |

## Selecting a kernel configuration by hand

The prefill launchers self-tune their launch configuration on the first call per
`(d, sq, skv, Hq, G)` shape and cache the winner. To see what they pick:

```bash
HIPDNN_PREFILL_TUNE_DEBUG=1 ./test_gqa_prefill.exe --iters 10
```

On Windows PowerShell, set it with `$env:HIPDNN_PREFILL_TUNE_DEBUG = '1'` and
clear it with `Remove-Item Env:\HIPDNN_PREFILL_TUNE_DEBUG`.
