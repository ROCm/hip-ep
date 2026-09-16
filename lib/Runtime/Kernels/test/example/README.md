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
```

`example/` is UT-only. The offline autotune LUT sweep tools (not UTs) live under
`hip/autotune/<op>/tools/` next to each op's real resolver — e.g.
`hip/autotune/gqa/tools/`, `hip/autotune/matmul_nbits/tools/`,
`hip/autotune/gemm/tools/`.

## Required files in every leaf — EXACTLY these five, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_shapes` / `test_model` / `test_custom` / `clean` |
| `test_<op>.cpp` | Driver. Inline empty `resolve()` if needed, guarded by `#ifndef HIPDNN_LUT_LINKED_EXTERNALLY` so `MODE=lut`'s real resolver (which defines the same symbols) can link. No `autotune_stub.cpp`. |
| `gen_data.py` | **The single data generator.** One file. Writes `data/` (inputs ± golden). `make test` runs it first. |
| `README.md` | Four targets + `MODE` |
| `shapes.csv` | Default shape list |

Allowed exception: `example/common/` (`csv_writer.h`, `embed_lut.py`) is shared
by every leaf's `test_<op>.cpp` / `Makefile` for `out/results.csv` and the
`MODE=lut` hex-embed — a small shared utility, not a second framework, so it
lives once instead of five times.

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

`test`, `test_shapes`, `test_model`, `test_custom`, `clean`.
CI: `cd example && make test OFFLOAD=... HIP_SDK=...`

### `MODE=auto` (default) — real LUT when one exists

`MODE ?= auto` in every leaf Makefile:

- **`auto`** (default): `lut` if `hip/autotune/<op>/lut/<arch>.fb` exists for
  the current `OFFLOAD` arch, else `autotune`. Never fails for lack of a
  table — falls back silently.
- **`lut`**: forces `lut`; if the `.fb` is missing for this arch, warns and
  falls back to `autotune` (still does not fail).
- **`autotune`**: always the runtime sweep, never touches a LUT.

`MODE=lut` needs `flatc` (a build tool, **not part of this repo** — see
Step 2 of the handoff that added this / `knowledge/hosts.md`) plus its
`include/`, passed on the command line:

```
make test MODE=auto OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> \
    FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<flatbuffers include dir>
```

`FLATC`/`FLATBUFFERS_INC` are only required when `MODE` actually resolves to
`lut` (checked at `make direct`/`$(TARGET)` time, not for `clean` etc.).
Under the hood: `flatc --cpp --gen-object-api --scoped-enums` generates
`<op>_autotune_generated.h` into `out/gen/`, `example/common/embed_lut.py`
hex-embeds `lut/<arch>.fb` into `out/gen/<op>_lut_data.cpp` as the same
`k<Op>LutData[]` symbol `tools/empty_lut_data.cpp` already uses (never
renamed), and the test links the real `hip/autotune/<op>/<op>_autotune.cpp`
resolver instead of the leaf's inline stub.

For matmul_nbits and gemm the kernel calls `resolve()` itself, so linking the
real resolver is enough. For GQA, `gqa_kernel.hip` does not call the resolver
(only production `real/gqa.cpp` does) — `test_gqa_decode.cpp` /
`test_gqa_prefill.cpp` instead call `hip_gqa_autotune_resolve_decode` /
`_resolve_prefill` themselves (guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`)
and dispatch the resolved config through the production
`hip_gqa_flash_decode_configured` / `_prefill_v3_configured` entries.

Fully wired end-to-end today: `matmul_nbits/Xfp16_Wu4_Yfp16`,
`gqa/decode/Xfp16_Wfp16_Yfp16`, `gqa/prefill/Xfp16_Wfp16_Yfp16`. The other six
leaves build and pass in both `MODE=auto`/`lut` and `MODE=autotune` (the real
resolver links cleanly), but their bit-width/dtype path does not call the
resolver today (matmul_nbits u2/u3/i8 — the LUT is bits=4-only; gqa decode/
prefill i8 dequantize to fp16 and reuse the unconfigured fp16 entry; gemm has
`MODE=lut` wired but not yet `out/results.csv`) — see each leaf's git history
for the exact scope of a follow-up.

### `out/results.csv`

Every `test` / `test_shapes` / `test_model` / `test_custom` on a leaf wired
for it appends to `out/results.csv` (truncated once per top-level invocation,
including across `test_shapes`'/`test_model`'s recursive per-shape calls;
`make clean` removes it with the rest of `out/`). Stable header:

```
op,leaf,arch,mode,shape,config,time_ms,gflops,gbps,is_best,lut_source,relL2,verdict
```

- **`autotune` mode**: one row per candidate config the tuner timed for that
  shape (`config` = the tuner's own log line, `is_best=1` on the winner — read
  off the `[custom_kernels]`/`[prefill-v5-tune]` debug lines the tuner already
  prints, captured by briefly redirecting stderr around the first, cache-cold
  call — see `example/common/csv_writer.h`), plus a final `config=final` row
  with `relL2`/`verdict`.
- **`lut` mode**: one row for the resolved config, `lut_source=exact|nearest`
  (or `fallback`/`heuristic`), `is_best=1`, its measured perf, `relL2`/
  `verdict`.
- The test also **prints** to stdout as before (the CSV is additive, not a
  replacement for the existing PASS lines).

Implemented via `example/common/csv_writer.h` (`CsvWriter`, `captureLogLines`,
`parseCandidateLines`) — env-driven (`HIPDNN_RESULTS_CSV/_OP/_LEAF/_ARCH/_MODE`,
set by each leaf's `Makefile`), so `test_<op>.cpp` needs no new CLI flags.
