<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# RyzenAI CoreLib (NPU) Backend

hip-ep can run selected operators on the AMD NPU through RyzenAI CoreLib
(`ryzenai-corelib`, C API in `include/ryzenai/corelib.h`). Every other operator stays on the GPU, and each
NPU-routed call falls back to the GPU at runtime when CoreLib cannot run it.

## Enabling

| Setting | Where | Effect |
|---|---|---|
| `use_ryzenai_corelib` = `1`/`true` | EP provider option | Compile with the CoreLib lowering |
| `HIPEP_USE_RYZENAI_CORELIB` = `1`/`true` | Environment, read at compile time | Same, when the provider option is unset |
| `RYZENAICORELIB_ROOT` | Environment, build and run time | CoreLib install tree (or repo root; `install/` is searched). Build: locates `include/ryzenai/corelib.h`. Run: loads `bin/ryzenai_corelib.dll` (`lib/libryzenai_corelib.so` on Linux), else the default library search path |
| `RYZENAICORELIB_PDI` = `<prefill>,<token>` | Environment, run time | PDI pair of the CoreLib stream; must match the model's kernels (see below). Default `0,16` |
| `RYZENAICORELIB_KERNEL_BUNDLE` = `<model>` | Environment, run time | Enables the kernel bundle linked into that CoreLib model library (e.g. `qwen3` from `ryzenai_corelib_model_qwen3.dll`) |
| `RYZENAI_CORELIB_KERNELS_ROOT` | Environment, run time (read by CoreLib) | NPU kernel store. With neither a store nor a bundle every call goes to the GPU |
| `HIPDNN_EP_STRICT` = `1` | Environment, run time | A call CoreLib ran on the host CPU fails inference with `HIPDNN_EP_ERR_CORELIB_CPU_FALLBACK` |

Both artifact formats are supported: `LLVM_IR` resolves the wrappers from the
host process, and `NATIVE` model DLLs link `hipdnn-ep-corelib`. An EPContext
artifact keeps the setting it was compiled with.

## Operator coverage

| HIP op | CoreLib call | Runs on CoreLib when |
|---|---|---|
| `hip.matmul_nbits` | `ryzenai_corelib_matmul_bf16` | `bits == 4`, no `g_idx`, `block_size` a multiple of 32 dividing K, zero points absent or packed uint8, and a kernel ships for (K, N, group, bias) |
| `hip.rms_norm` | `ryzenai_corelib_rmsnorm_bf16` | the scale spans exactly the normalized width, and a kernel ships for it |
| `hip.skip_rms_norm` | `ryzenai_corelib_add_rmsnorm_bf16` | no bias input, and a kernel ships for the width |

Activations may be f32, f16 or bf16; CoreLib computes in BF16. Weights are
packed once per op instance from the model constants. Attention (`hip.gqa`)
stays on the GPU: CoreLib's `flat_mha_bf16` owns its KV cache on the NPU,
which does not match ONNX Runtime GenAI's past/present tensors.

Each CoreLib model ships kernels for one PDI pair, and a stream on another pair
silently runs those shapes on the host (which hip-ep then routes to the GPU).
Set `RYZENAICORELIB_PDI` to the pair of the CoreLib model matching the ONNX
model, as declared in that model's `models/<Model>.cpp` (`kPrefillPdi`,
`kTokenPdi`). At CoreLib 0.29.5:

| CoreLib model | `RYZENAICORELIB_PDI` |
|---|---|
| Qwen2 / Qwen2.5 | `0,16` (default) |
| Qwen3, Llama | `1,16` |
| Gemma4 | `8,17` |
| Qwen3.5 | `9,19` |

Matmul group size and shipped row counts are read from CoreLib's kernel
enumeration, so they need no configuration. An ONNX int4 export at block size
64 or 128 reaches g64 kernels; block size 32 selects g32 kernels.

## Generated code

With `use-ryzenai-corelib`, `convert-hip-to-llvm` emits the CoreLib wrapper
first and guards the GPU wrapper on its result
(`lib/Conversion/HipToLLVM/RyzenAICorelibLowering.cpp`):

```mlir
%rc = llvm.call @wrap_corelib_matmul_nbits(%state, %slot, ..., %dtype)
%gpu = llvm.icmp "eq" %rc, %c1 : i32          // HIPDNN_EP_CORELIB_NOT_HANDLED
llvm.cond_br %gpu, ^gpu, ^cont
^gpu:
  llvm.call @wrap_matmul_nbits(%state, %slot, ...)
  llvm.br ^cont
```

Each `wrap_corelib_*` takes its GPU wrapper's exact arguments plus the
activation data type (`HIPDNN_EP_DATATYPE_*`), so the fallback needs no
argument translation.

## Runtime

`lib/Runtime/real/ryzenaicorelib.cpp` is compiled natively into the
`hipdnn-ep-corelib` static library (not `runtime.bc`) and reaches the JIT as
absolute symbols registered in `LlvmIrJit`, like the current-stream accessors.
When the CoreLib header is absent at build time, or in mock builds,
`lib/Runtime/mock/ryzenaicorelib_stub.cpp` takes its place and every call
returns `HIPDNN_EP_CORELIB_NOT_HANDLED`.

CoreLib is loaded with `LoadLibrary`/`dlopen` on first use rather than linked,
so the EP loads on machines without it. The library's version must match the
header's major (and, below 1.0, minor) version.

Per session (`RuntimeState::corelib_ctx`, released through
`corelib_ctx_destroy` in `hipdnn_ep_state_cleanup`):

- One CoreLib stream on the configured PDI pair, with the configured kernel
  bundle enabled. No NPU, no loadable library, a malformed setting, a bundle
  that fails to enable, or neither a bundle nor a kernel store makes the
  context unusable, and every call returns not-handled.
- Per op instance, keyed by its constant weight pointer (stable for the
  session), op kind and epsilon: the packed weights, the row counts that ship
  for the shape (from `*_enum_kernels`), and BF16 device tensors per padded
  row count.

Per call:

1. Rows (`M * batch`, or elements / width for the norms) are padded up to the
   smallest shipped row count; none means not-handled.
2. Inputs are copied GPU → host on the session stream, written into CoreLib
   tensors, run, synchronized, read back, and copied host → GPU. CoreLib
   exposes no device pointer, so this staging is unavoidable.
3. CoreLib's `cpu_fallbacks` counter is compared before and after. A host run
   produces a correct result; that padded row count is then routed to the GPU
   for the rest of the session, or, under `HIPDNN_EP_STRICT=1`, the device
   error flag is set so `inference_compute` fails.

The last session to close calls `ryzenai_corelib_cleanup`, which CoreLib
requires before process exit.

## Limitations

- Every routed call stages through host memory and synchronizes the GPU
  stream, so per-op offload is expected to be slower than the GPU path; it
  establishes the plumbing and numerical correctness.
- CoreLib kernels ship only at specific shapes; anything else falls back.
- Attention, the fused MLP (`ssmlp_bf16`) and RoPE are not routed.
