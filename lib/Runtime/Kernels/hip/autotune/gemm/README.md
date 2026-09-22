<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GEMM autotune / LUT

Offline-tuning home for the custom ONNX `Gemm` kernel (`lib/Runtime/Kernels/hip/gemm_kernel.hip`).
Mirrors the `matmul_nbits` autotune design (`../matmul_nbits/`), and as of
schema v2 shares its dtype vocabulary with every other op via `../common/`.

## Schema v2: explicit dtypes, four phases

v1 keyed a point on `type_bytes` (2/4/8) -- an element-size proxy that cannot
tell fp16 from bf16 (both 2 bytes), cannot express activation != weights, and
is meaningless for the sub-byte quantized weights `matmul_nbits` needs next
(see `plan.md` §1 D2 for the full design review). v2 replaces it with three
explicit fields from the shared `HipdnnDType` vocabulary
(`../common/hipdnn_dtype.fbs`): `act_dtype` / `wts_dtype` / `out_dtype`.
`out_dtype` is stored on every point but is **not** part of the grouping key
-- every point measured so far has `out == act`, and `out` only changes the
epilogue store, never the tile choice (promote it into the key later if a
measured `out != act` point ever changes a winner; that's a data change, not
a re-measurement).

The grouping key is `(phase, act_dtype, wts_dtype, trans_b)`, packed as an
8-bit-per-lane `uint64_t` (`groupKey()` / `fallbackKey()` in
`gemm_autotune.cpp`) so a phase or dtype enum growing past its old 2-bit field
can never silently collide with another group (v1's D3 bug).

Four phases now, not two:

| phase | serves | config kind |
|---|---|---|
| `Wmma` | prefill tiles, fp16/bf16 only | `Wmma` (bm16/bn16/swizzle/wt_m/wt_n/bk/split_k) |
| `GemvNt` | decode, N-major (the original GEMV path) | `Gemv` (threads/tile_n) |
| `GemvNn` | decode, N-parallel (previously a single hardcoded `BLOCK=256`, no table at all) | `Gemv` (threads only, `tile_n` unused/0) |
| `TiledFma` | the plain-FMA tiled path: fp32, fp64, and wave64 fp16/bf16 (no WMMA on wave64 hardware) | `TiledFma` (bm16/bn16/bk/wt_m/wt_n=TM/TN/threads) |

`phase -> config kind` is an explicit table (`phaseConfigKind()`, fixes v1's
D4 bug where a third phase would have been silently accepted as a `Gemv`
config through a `wmma ? Wmma : Gemv` boolean guess).

**`fallbacks` is always empty this round, on purpose** (D5 note in
`plan.md` §1): an empty fallback table plus exact-match-only groups is what
gives a dtype with no measured points a clean **miss** back to the static
default, instead of silently borrowing another dtype's (or another phase's)
point. `fallbackKey()` shares `groupKey()`'s bit layout and *does* carry
`trans_b` (v1 dropped it, conflating NN/NT winners), so filling `fallbacks`
in later is a pure data change whenever that's wanted.

## Config selection order (dispatch path)

Every phase's tuner (`tuneWmma` / `tuneGemvNt` / `tuneGemvNn` / `tuneTiledFma`
in `gemm_kernel.hip`) resolves a config in this order:

1. **Forced env** -- `HIPDNN_EP_GEMM_WMMA_CFG` / `HIPDNN_EP_GEMM_GEMV_CFG` /
   `HIPDNN_EP_GEMM_GEMV_NN_CFG` / `HIPDNN_EP_GEMM_TILEDFMA_CFG` pin one table
   index and skip everything below (profiling / A-B isolation).
2. **In-process map** -- winners measured on this exact machine+build, latched
   for the process lifetime. The cache key includes the dtype name (D1 fix:
   v1 keyed on `sizeof(T)`, so an fp16 GEMM and a bf16 GEMM on the same shape
   collided and the second dtype silently reused the first's config without
   ever calling `resolve()`).
3. **Offline LUT** (`lookup` mode only) -- `gemm_lut::resolve`. Stores tile
   *geometry*, not an index; a validator maps geometry back onto the live
   config table so a stale table is skipped rather than launched.
4. **Sweep** -- the in-process autotune sweep (`online` mode only, or a LUT
   miss in `lookup` mode for `Wmma`/`GemvNt`/`GemvNn`/`TiledFma` -- `TiledFma`
   with `transA=1` is the one exception: it is not measured or looked up this
   round, plan.md's Stage B matrix only covers `transA=0`, so a `transA=1`
   caller always keeps the pre-LUT default geometry).

## Mode switch

`HIPDNN_GEMM_AUTOTUNE_MODE` (read once, latched), same semantics as
`HIPDNN_MATMUL_AUTOTUNE_MODE`:

| value | behaviour |
|---|---|
| `lookup` (default) | resolve from the offline LUT; on a miss use the static occupancy heuristic (`pickWmmaHeuristic` / GEMV/TiledFma defaults), NOT the on-device sweep |
| `online` | bypass the LUT and run the on-device sweep (used to generate LUT data under controlled conditions) |

The `lookup`-miss path deliberately avoids the runtime sweep: on the gfx1151
APU compute-bound tiles throttle hard and the throttling correlates with tile
size, so an on-device sweep systematically mis-ranks configs (measured
picking 128x64 over the ~2x-faster 128x128 WT2x4 on large-K shapes). The
heuristic is robust; the sweep exists to populate the durable LUT offline.

`HIPDNN_EP_GEMM_AUTOTUNE=0` disables the sweep entirely and uses the static
occupancy heuristic, for perf isolation.

## Measurement protocol (plan.md §4 / `hip-kernel-perf-measurement` skill)

Every tuner (not just `tuneWmma`) now shares two helpers in `gemm_kernel.hip`:

- `gemmSettleClock()` -- drives a representative config until per-iter latency
  plateaus (>=60 ms total, then 3 consecutive reps within 2%) *before* the
  candidate loop starts. `tuneGemvNt` did not have this before this round
  (WARMUP=1, no settle, averaged, no negative-value guard) -- exactly the
  shape of the gfx1152 decode-tuner incident the skill documents (a table
  shipped from an unsettled sweep was slower than the tuner it replaced).
- `gemmTimeConfigPeak()` -- per-candidate WARMUP launches (untimed,
  re-equilibrate) then ITERS timed launches, keeping the **min** (peak, least
  throttled) with a negative-value guard (`hipEventElapsedTime` occasionally
  returns a small negative on sub-0.1 ms kernels, and an ungated min always
  "wins" with it).

The **offline tie-break** (near-ties within 3% preferring the larger warp
tile) lives in `scripts/update_lut.py`, not in the kernel -- the runtime
tuner stays a plain single pass; the more careful choice is made once,
offline, from the full per-config timings the sweep logs record (mirrors
`../matmul_nbits/scripts/update_lut.py`).

## Generating the table

```
# 1. Build the sweep binary (single-kernel hipcc, not a full CMake configure):
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -w \
    -I lib/Runtime/Kernels/include \
    lib/Runtime/Kernels/hip/autotune/gemm/tools/gemm_autotune_sweep.cpp \
    lib/Runtime/Kernels/hip/gemm_kernel.hip \
    lib/Runtime/Kernels/hip/autotune/gemm/tools/gemm_autotune_stub.cpp \
    -o gemm_sweep.exe

# 2. Measure (drives the sweep, writes scripts/data/<arch>[_<tag>]_sweep.log):
python scripts/update_lut.py measure --arch gfx1151 --sweep ../gemm_sweep.exe \
    --shapes shapes/gemm_shapes.csv

# 3. Build (merges EVERY scripts/data/<arch>*_sweep.log; a chunked or
#    targeted --tag remeasure pass never needs a full re-sweep):
python scripts/update_lut.py build --arch gfx1151

# 4. Compile (needs flatc + its include/, a build tool, not part of this repo):
python scripts/update_lut.py compile --arch gfx1151 --flatc <flatc>
```

`measure` accepts `--m`, `--dtypes`, `--phases`, `--limit` to run a subset
(the sweep binary itself takes the same flags directly -- see
`tools/gemm_autotune_sweep.cpp`'s header comment for its full CLI and
`--max-mem-frac`, which replaces a hardcoded byte cap so large-N shapes like
a vocab projection are not silently skipped for every dtype).

`build` also writes `scripts/data/unstable_shapes.csv` for any point whose
winner and runner-up were within 3% (plan.md §4.3) -- re-measure those with a
targeted `measure --tag remeasure ...` before trusting them; two rounds that
still disagree should stay out of the table (fall back to the static
default) rather than being forced in.

**After `compile`, re-configure (not just rebuild).** CMake embeds the `.fb`
bytes into the DLL at configure time (`file(READ ... HEX)`); a build-only
refresh leaves the DLL on the old bytes. `HIPDNN_GEMM_LUT_LOG=1` (or
`HIPDNN_EP_DEBUG=1`) confirms which table actually loaded.

## bf16 vs fp16: merge decision (plan.md §5 Stage C)

**Measured 2026-09-19, verdict: do not merge (`bf16_aliases_f16 = false`).**
Every Wmma shape with both an f16 and a bf16 measured point (2968 shapes)
was compared by winning config index (same global config pool, so an equal
index means an identical tile geometry):

- **Only 62.84% (1865/2968) picked the identical winner config** -- far
  below the >=95% bar plan.md §5 Stage C sets for merging.
- For the 1103 shapes that picked a *different* config, the cost of forcing
  the other dtype's winner (measured from the same shape's own candidate
  timings, not estimated): median ~2-3% either direction, but the tails are
  large -- up to **457% slower** forcing bf16's winner onto f16, and up to
  **2000% slower** forcing f16's winner onto bf16 (n=293 shapes where both
  candidates' timings were available in the raw sweep log; the coarse
  autotune pass does not try every config on every shape, so a cross cost
  could not be computed for every differing shape, but the sample is large
  enough that the tails are not a fluke).

**The old undocumented assumption ("f16 and bf16 share tile geometry" --
`gemm_autotune.fbs:26`'s original comment) was wrong.** bf16 gets its own
measured group in the table, exactly like fp16 and fp32; `resolve()` never
aliases a BF16 query to an F16 point. Analysis script and raw output:
`handoffs/gemm_lut_dtype_gfx1151/artifacts/stage_c_analysis.py` /
`stageC_bf16_vs_f16_analysis.txt` on 171.

## Distance weights (plan.md §1 D8)

`weight_m` / `weight_n` / `weight_k` stay fixed at `1.0` this round -- a
single weight triple for the whole table, same as v1. **Enabling weight
fitting requires first changing this to per-group weights**: fp16 prefill
(all three of M/N/K vary) and fp32 decode (M is pinned at 1, so `log2(M)` is
a constant) do not belong in the same distance metric. `GemmAutotuneLut`
adding a `[GemmGroupWeights]` table is a flatbuffers field-add (backward
compatible, no `schema_version` bump needed) -- do that before fitting
weights, not after.

## Status

Infrastructure is **wired into the DLL** (mirrors `../matmul_nbits/`):

- `gemm_autotune.h/.cpp/.fbs` -- API, flatbuffer-backed nearest-neighbour
  resolver (arch/schema/ABI gated), schema v2 (`schema_version=2`,
  `kernel_abi="gemm-v2"` -- an old v1 `.fb` is rejected at load, falls back to
  the static default, and every leaf still passes; this is expected, R11).
- `../common/hipdnn_dtype.{fbs,h}` + `dtype.py` -- the shared dtype
  vocabulary (one enum, three consumers: this schema, the C++ loader, and
  `update_lut.py`; append-only, never renumber).
- `gemm_kernel.hip` includes the header and aliases `gemm_lut =
  hipdnn_ep::gemm_autotune`; the map -> LUT -> sweep dispatch and the
  `wmmaConfigFromGeom` / `gemvNtConfigFromGeom` / `gemvNnConfigFromGeom` /
  `tiledFmaConfigFromGeom` validators call `resolve()`.
- `schemas/CMakeLists.txt` generates `gemm_autotune_generated.h` (flatc), and
  `lib/Runtime/Kernels/CMakeLists.txt` links `gemm_autotune.cpp` and embeds the
  checked-in `lut/<arch>.fb` bytes into each `custom_kernels_<arch>` with pure
  CMake (`file(READ ... HEX)`) -- no generated `.cpp` is committed. Mirrors
  `../matmul_nbits/` and `../gqa/`.

Only the `.fb` (plus its `.json` source) ships. An arch with **no** measured
table still builds: `resolve()` reports a miss, and every shape falls through
to the heuristic / runtime sweep. Landing or growing a table is therefore a
pure data addition (a `.fb`), never a code or build change.

## Layout

```
gemm/
  gemm_autotune.h        # LUT lookup API (Request / WmmaAnswer / GemvAnswer / TiledFmaAnswer / resolve)
  gemm_autotune.cpp      # flatbuffer-backed resolver, arch/schema/ABI gated, uint64 group keys
  gemm_autotune.fbs      # schema v2 (includes ../common/hipdnn_dtype.fbs)
  shapes/gemm_shapes.csv       # 317 (N,K,category,priority,source,m_list) rows, derived from HF configs
  shapes/hf_model_configs.json # provenance for gemm_shapes.csv
  scripts/derive_shapes.py     # regenerates gemm_shapes.csv from hf_model_configs.json
  scripts/update_lut.py  # measure (drive the sweep) + build (logs -> json) + compile (json -> fb)
  scripts/data/          # gitignored raw sweep logs -- one file per measure/tag, never committed
  lut/gfx1151.json       # human-readable source (built from scripts/data logs)
  lut/gfx1151.fb         # committed table; CMake embeds its bytes into the DLL
  tools/gemm_autotune_sweep.cpp # offline sweep driver: touches (shape,dtype,phase) once, lets the
                                # kernel's own tuner log its winner (one implementation of "who wins")
  tools/gemm_autotune_stub.cpp  # flatbuffers-free resolve() for the standalone Makefile / the sweep tool
```
