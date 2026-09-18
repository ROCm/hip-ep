<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GEMM autotune / LUT

Offline-tuning home for the custom ONNX `Gemm` kernel (`lib/Runtime/Kernels/hip/gemm_kernel.hip`).
Mirrors the `matmul_nbits` autotune design (`../matmul_nbits/`).

## Config selection order (dispatch path)

Both the WMMA prefill path and the GEMV NT decode path resolve a config in this
order (see `tuneWmma` / `tuneGemvNt` in `gemm_kernel.hip`):

1. **Forced env** — `HIPDNN_EP_GEMM_WMMA_CFG` / `HIPDNN_EP_GEMM_GEMV_CFG` pin one
   table index and skip everything below (profiling / A-B isolation).
2. **In-process map** — winners measured on this exact machine+build, latched for
   the process lifetime.
3. **Offline LUT** — `gemm_lut::resolve` (only in `lookup` mode). Stores tile
   *geometry*, not an index; a validator maps geometry back onto the live config
   table so a stale table is skipped rather than launched.
4. **Sweep** — the in-process autotune sweep (only in `online` mode, or on a LUT
   miss in `lookup` mode).

## Mode switch

`HIPDNN_GEMM_AUTOTUNE_MODE` (read once, latched), same semantics as
`HIPDNN_MATMUL_AUTOTUNE_MODE`:

| value | behaviour |
|---|---|
| `lookup` (default) | resolve from the offline LUT; on a miss use the static occupancy heuristic (`pickWmmaHeuristic` / GEMV default), NOT the on-device sweep |
| `online` | bypass the LUT and run the on-device sweep (used to generate LUT data under controlled conditions) |

The `lookup`-miss path deliberately avoids the runtime sweep: on the gfx1151 APU
compute-bound tiles throttle hard and the throttling correlates with tile size,
so an on-device sweep systematically mis-ranks configs (measured picking 128x64
over the ~2x-faster 128x128 WT2x4 on large-K shapes). The heuristic is robust;
the sweep exists to populate the durable LUT offline.

`HIPDNN_EP_GEMM_AUTOTUNE=0` disables the sweep entirely and uses the static
occupancy heuristic (`pickWmmaHeuristic`), for perf isolation.

## Status

Infrastructure is **wired into the DLL** (mirrors `../matmul_nbits/`):

- `gemm_autotune.h/.cpp/.fbs` — API, flatbuffer-backed nearest-neighbour
  resolver (arch/schema/ABI gated), schema.
- `gemm_kernel.hip` includes the header and aliases `gemm_lut =
  hipdnn_ep::gemm_autotune`; the map -> LUT -> sweep dispatch and the
  `wmmaConfigFromGeom` / `gemvNtConfigFromGeom` validators call `resolve()`.
- `schemas/CMakeLists.txt` generates `gemm_autotune_generated.h` (flatc), and
  `lib/Runtime/Kernels/CMakeLists.txt` links `gemm_autotune.cpp` and embeds the
  checked-in `lut/<arch>.fb` bytes into each `custom_kernels_<arch>` with pure
  CMake (`file(READ ... HEX)`) -- no generated `.cpp` is committed. Mirrors
  `../matmul_nbits/` and `../gqa/`.

Only the `.fb` (plus its `.json` source) ships. An arch with **no** measured
table still builds: CMake links `tools/empty_lut_data.cpp` (size 0), `resolve()`
reports a miss, and every shape falls through to the heuristic / runtime sweep.
Landing / growing a table is therefore a pure data addition (a `.fb`), never a
code or build change. `lut/gfx1151.fb` currently carries a small seed table.

## Layout

```
gemm/
  gemm_autotune.h        # LUT lookup API (Request / WmmaAnswer / GemvAnswer / resolve)  [done]
  gemm_autotune.cpp      # flatbuffer-backed resolver, arch/schema/ABI gated             [done]
  gemm_autotune.fbs      # schema                                                        [done]
  lut/gfx1151_points.csv # measured winners, source for the json                         [source]
  lut/gfx1151.json       # human-readable source (built from the CSV)                     [source]
  lut/gfx1151.fb         # committed table; CMake embeds its bytes into the DLL           [committed]
  scripts/update_lut.py  # build (CSV -> json) + compile (json -> fb)                     [done]
  tools/empty_lut_data.cpp     # size-0 payload for the no-table build                   [done]
  tools/gemm_autotune_stub.cpp # flatbuffers-free resolve() for the standalone Makefile  [done]
```

## Generating the table (next step)

1. Run a **cooled** per-shape sweep (default routing, cooldown between shapes --
   NOT the online-autotune loop, whose timing is thermally corrupted; see the
   perf skill) over `shapes/oga_models.csv`, choosing the trustworthy winner per
   shape, and write a points CSV
   (`phase,type_bytes,trans_b,M,N,K,bm,bn,wt_m,wt_n,swizzle,split_k,bk,threads,tile_n`).
2. `python scripts/update_lut.py build --csv lut/gfx1151_points.csv` -> `lut/gfx1151.json`.
3. `python scripts/update_lut.py compile --flatc <flatc>` -> `lut/gfx1151.fb` (commit this).
4. Rebuild: CMake embeds the `.fb` bytes into the DLL; `HIPDNN_GEMM_LUT_LOG=1` confirms the load.
