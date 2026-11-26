<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MIGraphX Backend and Whole-Graph Backend Selection — Design

**Date:** 2026-11-26
**Document Type:** Design
**Status:** Draft (compile-checked; not yet run on a GPU)

---

## 1. Overview

hip-ep can run a session's graph through one of two backends:

- **HIP** — the hip-ep compiler (MorphiZen → MLIR → LLVM bitcode). Unchanged.
- **MIGraphX** — the MIGraphX library parses the ONNX model file and compiles
  it for the GPU.

The choice is made once per session, in `GetCapability`, for the **whole
graph**. A session never mixes backends, and the MIGraphX path never splits a
graph: it claims every node or none.

```
ORT GetCapability(graph)
   │
   ├─ gpu_backend=migraphx ──────────────► MIGraphX: eligible? compiled? ─► claim all / claim none
   │
   ├─ gpu_backend=hip ───► HIP compile ──► claimed by HIP / claim none
   │
   └─ gpu_backend=auto ──► HIP compile ──► claimed by HIP
          (default)             │
                                └─ HIP claimed nothing ─► MIGraphX: eligible? compiled? ─► claim all / claim none
```

"Claim none" leaves the nodes to the next EP (normally ORT's CPU EP). With
`session.disable_cpu_ep_fallback=1`, session creation fails instead.

## 2. Provider option

| Key | Values | Default |
|---|---|---|
| `gpu_backend` | `hip`, `migraphx`, `auto` | `auto` |

Values are exact and case-sensitive. Any other value makes EP creation fail
with `invalid provider option gpu_backend='<value>'; expected hip, migraphx or
auto`.

`migraphx` never falls back to HIP: it is the switch used to measure the
MIGraphX path on its own. `auto` behaves exactly like `hip` when HIP compiles
the graph, so the default is unchanged for every graph HIP supports today.

## 3. Routing policy v1

Source: `morphizen/ort-bridge/src/backend-routing.{hpp,cpp}`
(`kRoutingPolicyVersion = 1`).

1. HIP first. If HIP compiles the graph, HIP runs it.
2. Otherwise MIGraphX, if the graph is **eligible** and MIGraphX compiles it.
3. Otherwise no claim.

MIGraphX eligibility (`migraphx_eligibility`):

| Rule | Why |
|---|---|
| The session has a model file path | The plugin parses the file; a session created from bytes has no path |
| Every graph input has static dims | MIGraphX compiles for one shape |
| Every graph output has static dims | Outputs are allocated before the run; there is no in-graph output allocation |
| No `GroupQueryAttention`, `PagedAttention`, `SparseAttention`, `LSTM`, `GRU`, `RNN` | State that persists across runs stays with HIP, which owns KV caches today |
| No graph input whose name starts with `past_` | Same: a KV-cache input |

Every decision is logged at ORT INFO level and through `MY_LOG(1)`:

```
hip-ep backend=migraphx policy_version=1 requested=auto reason=HIP did not compile the graph
hip-ep backend=none policy_version=1 requested=migraphx reason=MIGraphX unavailable: <error>
```

Any change to these rules bumps `kRoutingPolicyVersion` and updates
`test/runtime/test_backend_routing.cpp`.

## 4. Components and contracts

```
libhipgpu.so (EP)                                  hipep_migraphx.so (plugin)
  morphizen-ep.cpp                                   lib/Backend/MIGraphX/
    GetCapability ─ backend-routing ─┐                 hipep_migraphx_plugin.cpp
    Compile / Compute                │                   links migraphx::c
  migraphx-backend.cpp ──────────────┴── dlopen ──►    implements migraphx-plugin-abi.hpp
    MigraphxProgram                     (C ABI v1)
```

- **Only the plugin links MIGraphX.** `libhipgpu` loads
  `hipep_migraphx.so` / `hipep_migraphx.dll` from its own directory on first
  use. When the plugin is missing or cannot load (for example MIGraphX's own
  libraries are not on the loader path), the MIGraphX path reports
  "unavailable" and `auto` keeps today's behaviour.
  `HIPDNN_EP_MIGRAPHX_PLUGIN_DIR` overrides the directory (tests use it to
  load a mock plugin).
- **C ABI** (`morphizen/ort-bridge/src/migraphx-plugin-abi.hpp`,
  `HIPEP_MGX_PLUGIN_ABI_VERSION 1`): compile, output count, output shape, run,
  release. No MIGraphX or ORT type crosses it. Errors are returned as an int
  plus a message buffer; no exception crosses it.
- **Compile** (during `GetCapability`, so a failure can still fall back):
  `parse_onnx(model_path)` with the static input shapes from the ORT graph,
  `offload_copy=false` (bind ORT's device buffers directly), `fast_math=false`
  (match ORT's MIGraphX EP numerics), target `gpu`. The plugin records the
  program's output parameters `<module>:#output_<i>`; their count must equal
  the graph's output count.
- **Claim**: every node in one fused node, with
  `drop_constant_initializers=true` (MIGraphX reads weights from the file).
- **Compute**: check each input shape equals the compiled shape; allocate
  outputs with `KernelContext_GetOutput`; bind inputs by name and outputs by
  index; `eval`; then `finish()` on the MIGraphX context, because hip-ep is not
  stream-aware and ORT reads outputs as soon as `Compute` returns. A mutex
  serializes runs of one compiled program. Errors return an `OrtStatus`.

## 5. Build

```
python build.py ... --cmake_extra_defines HIPDNN_EP_ENABLE_MIGRAPHX=ON \
                    --cmake_prefix_path "<ORT>;<MIGRAPHX_SDK>"
```

`HIPDNN_EP_ENABLE_MIGRAPHX` (default `OFF`) adds `lib/Backend/MIGraphX`, which
needs `find_package(migraphx)`. With the option off nothing about the build
changes; `libhipgpu` never links MIGraphX either way.

## 6. Decisions and deviations from the execution plan

| Topic | Decision | Reason |
|---|---|---|
| ONNX source | Model file path (plan option B) | No protobuf export; follows the "no raw ONNX protobuf" rule. Sessions from bytes are ineligible |
| Default | `auto` | Same result as `hip` for every graph HIP compiles |
| Linking | Separate plugin, loaded at run time | `libhipgpu` builds and runs where MIGraphX is absent; keeps MIGraphX's LLVM/COMGR out of the EP's link |
| Policy storage | Constant in code (`kRoutingPolicyVersion`) | No benchmark data exists yet for a data-driven table |
| Execution | Synchronous `eval` + `finish` | hip-ep is not stream-aware; `run_async` on an EP stream comes with stream support |
| HIP path | Not moved behind a backend interface | One `if` in `GetCapability`/`Compile` is enough for two backends |
| EPContext | Not supported on the MIGraphX path (`ORT_NOT_IMPLEMENTED`) | Needs an `.mxr` save/load design |

## 7. Not in this change (open items)

- **GPU validation.** Not yet run on gfx1151 or MI300. The plugin has been
  compiled against MIGraphX headers (AMDMIGraphX `20bdaa7`) and its imports
  resolve against `libmigraphx_c.so.3` from the `migraphx_libs 2.18.0+rocm10.1.0`
  wheel, which matches hip-ep's TheRock 10.1.0.
- **Device selection.** Both backends still use HIP device 0.
- **Dynamic shapes**, quantization options, `.mxr` caching.
- **Accuracy and performance gates** (planned: cosine ≥ 0.999 and
  `allclose(atol=1e-3, rtol=1e-3)` against ORT CPU; MIGraphX path P50 within 5%
  of standalone MIGraphX). Not enforced by tests yet.
- **Packaging and CI**: shipping the plugin and MIGraphX runtime libraries in
  the wheel; a CI job with `HIPDNN_EP_ENABLE_MIGRAPHX=ON`.
- **Coexistence** of the HIP and MIGraphX runtimes (LLVM, COMGR) in one process
  must be checked on hardware.

## 8. Tests

| Test | Needs GPU | Checks |
|---|---|---|
| `BackendRoutingUnitTest` (`test/runtime/test_backend_routing.cpp`) | No | Option parsing; every eligibility rule |
| `MlirE2ETest.InvalidGpuBackendFailsSession` | No (mock) | A misspelt value fails session creation and names `gpu_backend` |
| `MlirE2ETest.MigraphxWithoutPluginClaimsNothing` | No (mock) | `gpu_backend=migraphx` never falls back to HIP |
| `MlirE2ETest.TwoLayerConvSession` | No (mock) | Default `auto` still routes a HIP-supported graph to HIP |
| `MigraphxMockE2ETest` (`backend-mlir-compiler/test/test_e2e_migraphx_mock.cpp`) | No (mock runtime + mock plugin) | Whole-graph claim and input/output binding; `auto` picks MIGraphX only when HIP fails; `hip` never uses MIGraphX; each eligibility rule on a real ORT graph; in-memory sessions; CPU fallback unchanged; compile failure, output-count mismatch, run failure, EPContext rejection |

`MigraphxMockE2ETest` loads `mock_migraphx_plugin.cpp`, which implements the
plugin ABI without MIGraphX and writes
`output[j] = sum_i (i + 1) * input[i] + 100 * (j + 1)`, so each test can tell
which backend produced a result and how buffers were bound. Its models come
from `backend-mlir-compiler/test/gen_migraphx_routing_models.py`. Like
`MlirE2ETest`, it is run directly (`test-e2e-migraphx-mock`); CTest does not
register tests under `backend-mlir-compiler/` because `enable_testing()` runs
after that directory is added.
