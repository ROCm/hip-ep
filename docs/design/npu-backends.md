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
