# Kernel unit tests (`lib/Runtime/Kernels/test/example`)

This folder is the **only** entry for standalone HIP kernel unit tests.
Makefile is the only driver. No `.bat` / `setup_*` / `run_*` wrappers.
No full-EP cmake.

Repo-relative paths only. `HIP_SDK` and `OFFLOAD` come from the command line.

## Leaf name = tensor dtypes, not a single "fp16"

A leaf folder is **one (X, W, Y) dtype combo**, plus optional suffixes for
storage/quant extras.

```
X<xdtype>_W<wdtype>_Y<ydtype>[_ZP<zpdtype>][_other]
```

| Token | Meaning |
|---|---|
| `X` | activation / input (GEMM A, GQA Q) |
| `W` | weight / KV (GEMM B, MatMulNBits packed B, GQA K/V cache) |
| `Y` | output (GEMM C, GQA O) |
| `_ZP…` | only if zero-point storage is a distinguishing feature of this leaf (`_ZPu8`, `_ZPfp16`, …) |

Examples:

| Leaf | What it is |
|---|---|
| `gemm/Xfp16_Wfp16_Yfp16` | dense GEMM fp16 |
| `matmul_nbits/Xfp16_Wu4_Yfp16` | A fp16, B uint4, C fp16 |
| `matmul_nbits/Xfp16_Wu2_Yfp16` | bits=2 |
| `matmul_nbits/Xfp16_Wi8_Yfp16` | unpacked int8 weights |
| `gqa/decode/Xfp16_Wfp16_Yfp16` | Q/O fp16, KV cache fp16 |
| `gqa/decode/Xfp16_Wi8_Yfp16` | Q/O fp16, KV cache int8 |
| `gqa/prefill/Xfp16_Wfp16_Yfp16` | same for prefill |
| `gqa/prefill/Xfp16_Wi8_Yfp16` | prefill INT8 KV |

If input becomes fp8 later: add `Xfp8_Wfp16_Yfp16`, do **not** rename the
existing fp16 leaf to a bare `fp16/` or `fp8/`. A folder named `fp16` or `i8`
is illegal — it does not say which tensor.

Do **not** use `fp16u4`, `decode/fp16`, `prefill/i8`.

## Layout

```
example/
  README.md
  Makefile                  # CI aggregator
  tests.manifest
  models/                   # shared JSON only
  gemm/Xfp16_Wfp16_Yfp16/
  matmul_nbits/Xfp16_Wu{2,3,4}_Yfp16/
  matmul_nbits/Xfp16_Wi8_Yfp16/
  gqa/decode/Xfp16_Wfp16_Yfp16/
  gqa/decode/Xfp16_Wi8_Yfp16/
  gqa/prefill/Xfp16_Wfp16_Yfp16/
  gqa/prefill/Xfp16_Wi8_Yfp16/
  gqa/autotune/             # LUT sweep, not a UT
```

## Required files in every leaf — EXACTLY these five, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_shapes` / `test_model` / `test_custom` / `clean` |
| `test_<op>.cpp` | Driver. Inline empty `resolve()` if needed. No `autotune_stub.cpp`. |
| `gen_data.py` | **The single data generator.** One file. Writes `data/` (inputs ± golden). `make test` runs it first. |
| `README.md` | Four targets + `MODE` |
| `shapes.csv` | Default shape list |

**One generator per leaf.** `gen_data.py` handles every mode itself via flags:

```
python gen_data.py SIZE=... GS=...          # single shape  (test / test_custom)
python gen_data.py --shapes shapes.csv ...  # shape list    (test_shapes)
python gen_data.py --model <json> ...        # model sweep   (test_model)
```

Do **not** ship a second/third generator. Forbidden filenames in a leaf:
`gen_matmul_nbits*_data.py`, `gen_model_data*.py`, `bench_*.py`, `verify_*.py`,
`zp_perf_compare.md`, `autotune_stub.cpp`, `test_model/*.json`, any committed
`out/` / `data/`, any home path. If a leaf has a real extra need, justify it in
its README — default is the five files above and nothing more.

GQA must not generate tensors only in C++. `gen_data.py` writes Q/K/V (and i8 KV +
scales/zp); the cpp loads `data/` then checks vs CPU/python golden.

`BUILD_DIR=out`. Model JSON lives only in `example/models/`.

## Uniform make API / CI

Same as before: `test`, `test_shapes`, `test_model`, `test_custom`,
`MODE=autotune|lut` (missing LUT → autotune, do not fail).
CI: `cd example && make test OFFLOAD=... HIP_SDK=...`
