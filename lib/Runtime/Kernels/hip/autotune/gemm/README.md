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

The LUT resolver is currently a stub (`gemm_lut::resolve` in `gemm_kernel.hip`)
that always reports a miss, so `lookup` behaves like `online` until a table is
generated. The dispatch hooks, geometry<->index bridges (`wmmaConfigFromGeom`,
`gemvNtConfigFromGeom`) and their validators are already wired, so embedding a
real table does not touch call sites.

## Planned layout (to fill)

Following `../matmul_nbits/`:

```
gemm/
  gemm_autotune.h        # LUT lookup API (Request / WmmaAnswer / GemvAnswer / resolve)
  gemm_autotune.cpp      # flatbuffer-backed resolver, arch/schema/ABI gated
  gemm_autotune.fbs      # schema
  lut/gfx1151.fb         # embedded table (per arch)
  lut/gfx1151.json       # human-readable source
  shapes/*.csv           # shape lists to sweep
  scripts/update_lut.py  # sweep -> json -> fb
  tools/                 # sweep / query / test harnesses
```

When this lands, replace the inline `gemm_lut` stub in `gemm_kernel.hip` with an
include of `gemm_autotune.h` and forward `resolve` to the embedded table (as
`matmul_lut` does), and register the `.cpp` in
`lib/Runtime/Kernels/CMakeLists.txt`.
