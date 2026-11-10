<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# NPU Backends

hip-ep can hand parts of a model to an AMD NPU through three optional
backends. All three build into the same `hipgpu.dll`, all are off by default,
and each loads its NPU library at run time, so the EP still loads, and runs
everything on the GPU, on machines without that library or NPU.

| Backend | Enable with | What goes to the NPU | NPU library | Design |
|---|---|---|---|---|
| CoreLib model-level | `use_ryzenai_corelib_model` / `HIPEP_USE_RYZENAI_CORELIB_MODEL` | A whole onnxruntime-genai decoder that a CoreLib model (now Qwen3) accepts, KV cache included | `ryzenai_corelib` | [ryzenai-corelib-model.md](ryzenai-corelib-model.md) |
| CoreLib per-op | `use_ryzenai_corelib` / `HIPEP_USE_RYZENAI_CORELIB` | MatMulNBits, RmsNorm and SkipRmsNorm, each falling back to the GPU when CoreLib cannot run it | `ryzenai_corelib` | [ryzenai-corelib-backend.md](ryzenai-corelib-backend.md) |
| DynamicDispatch | `use_dynamic_dispatch` / `HIPEP_USE_DYNAMIC_DISPATCH` | f16 GEMM and Conv of the few shapes DD ships kernels for | `dyn_dispatch_core` | [dynamicdispatch-integration.md](dynamicdispatch-integration.md) |

The first name in each "Enable with" entry is the EP provider option; the
second is the environment variable read when the option is unset.

## Building

Each backend needs only its library's headers at build time:
`RYZENAICORELIB_ROOT` names a CoreLib 0.66 or later install (both CoreLib
backends), and `DYNAMICDISPATCH_ROOT` a DynamicDispatch tree with its C API.
Without them, or in mock builds, stubs take their place: the CoreLib model path
refuses every model, CoreLib per-op calls report "not handled" and run on the
GPU, and DynamicDispatch calls fail.

The two CoreLib backends share one loader (`lib/Runtime/real/corelib_loader.h`):
one function table, one version check, and one count of live streams and model
sessions, so `ryzenai_corelib_cleanup` runs only after the last of either kind
is gone.

## When more than one is enabled

- The CoreLib model-level path decides first, in the level-1 pass, before any
  MLIR is built. When it takes the graph, nothing else compiles it; when it
  refuses, the graph compiles as usual with whatever per-op backends are on.
- CoreLib per-op and DynamicDispatch act in HipToLLVM on different operations
  today (MatMulNBits and RMS norms vs. f16 GEMM and Conv), so both can be on at
  once.
- Where they come to overlap, CoreLib takes precedence. CoreLib is where NPU
  development continues, including support for AIE2 NPUs, which will let it
  take over what DynamicDispatch runs today.

## Hardware

CoreLib's ops and models currently run on AIE4 NPUs (`ryzenai_corelib_npu_usable()`
is false elsewhere, and both CoreLib paths then step aside).
DynamicDispatch runs where its xclbins and XRT do. The per-op paths stage each
call's tensors through host memory; the model path keeps the weights and KV
cache on the NPU and moves only token ids and logits.

## Environment variables

Every backend can be enabled and located through the environment as well as
through provider options; the provider option wins where both exist.

### hip-ep: CoreLib whole-model path

| Variable | Read at | Effect |
|---|---|---|
| `HIPEP_USE_RYZENAI_CORELIB_MODEL` = `1`/`true` | session creation | Turns the path on when the `use_ryzenai_corelib_model` provider option is unset |
| `RYZENAICORELIB_ROOT` | build and run | Build: where CoreLib's headers are (`include/ryzenai/corelib/base.h`, 0.66 or later). Run: `bin/ryzenai_corelib.dll` (or `install/bin/`), else the default library search path |
| `RYZENAICORELIB_MODELS` | session creation | Comma-separated CoreLib model names to try instead of the model libraries found beside CoreLib, for built-in models or to narrow discovery. Added by #1476's model-agnostic matching, not yet merged into this branch |
| `RYZENAICORELIB_REPORT` = `1` | session end | Prints `[corelib] model report: ...` with the forwards, prefills, decodes and rows run on the NPU |

### hip-ep: CoreLib per-op path

| Variable | Effect |
|---|---|
| `HIPEP_USE_RYZENAI_CORELIB` = `1`/`true` | Tries CoreLib before the GPU for MatMulNBits, RmsNorm and SkipRmsNorm when the `use_ryzenai_corelib` provider option is unset |
| `RYZENAICORELIB_ROOT` | As above; both CoreLib paths share one loader |
| `RYZENAICORELIB_PDI` | `<prefill PDIs>;<decode PDIs>`, each a priority list (default `1;16`): the PDIs whose kernels the stream may use |
| `RYZENAICORELIB_KERNEL_BUNDLE` | The CoreLib kernel bundle new streams enable |
| `RYZENAICORELIB_REPORT` = `1` | At session end, what ran on the NPU and what fell back to the GPU |
| `HIPDNN_EP_STRICT` = `1` | hip-ep's general strict mode, which the per-op wrappers also honor |

### hip-ep: DynamicDispatch

| Variable | Effect |
|---|---|
| `HIPEP_USE_DYNAMIC_DISPATCH` = `1`/`true` | Lowers the supported GEMM/Conv shapes to DynamicDispatch when the `use_dynamic_dispatch` provider option is unset |
| `DYNAMICDISPATCH_ROOT` (or `DD_ROOT`) | Build: where `ops/dd_c_api.h` is found. Run: where `dyn_dispatch_core.dll` is loaded from, else the default library search path |
| `HIPEP_DD_DEBUG_NO_GET_OP_STATE` = `1` | Recreates each DynamicDispatch op on every call instead of caching it |

### Related, outside the NPU backends

| Variable | Effect |
|---|---|
| `MORPHIZEN_DEFAULT_HIPMALLOC` = `0` | The EP's default allocator returns host-mapped memory instead of device-only. Needed only with an onnxruntime-genai older than 0.16, whose AMDGPU device writes EP memory from the host (see [ryzenai-corelib-model.md](ryzenai-corelib-model.md)); slower for GPU-only runs |
| `AMDGPU_EP_PATH` | Where onnxruntime-genai loads `amdgpu-ep.dll` (the AMD GPU umbrella EP) from |

### CoreLib's own

`ryzenai_corelib.dll` reads these itself, whoever loads it; CoreLib's README
and `docs/` are authoritative. The ones that matter with hip-ep:

| Variable | Effect |
|---|---|
| `RYZENAI_CORELIB_DISABLE_GPU` = `1` | Keeps new CoreLib streams off the GPU. The model path already asks for the NPU and CPU only |
| `RYZENAI_CORELIB_KERNELS_ROOT` | A kernel store for new streams (per-op path; models link their own kernels) |
| `RYZENAI_CORELIB_PROFILE` | A profiling report when a stream is released (`1`, or a JSON file name) |
| `RYZENAI_CORELIB_TRACE` | Tracing, e.g. `xrt` |
| `RYZENAI_CORELIB_REPORT_FALLBACKS` = `1` | Host and GPU time per artifact |

Its others are for its tools and tests: `RYZENAI_CORELIB_PATH` (the Python
bindings' DLL), `RYZENAI_CORELIB_MODEL_CACHE` (downloaded models),
`RYZENAI_CORELIB_GPU_CHECK`, `_GPU_SKIP`, `_GPU_GATHER`, `_REQUIRE_GPU`,
`_SAFE_MODE`, `_XRT_UC_LOG`, `_QWEN36_NO_SUPER_KERNELS`, `_REFERENCE_CACHE`
and the `_STRESS_*` settings.
