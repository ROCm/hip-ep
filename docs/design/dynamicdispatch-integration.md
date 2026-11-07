<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# DynamicDispatch NPU Backend

DynamicDispatch (DD) runs a few GEMM and Conv shapes on the NPU through XRT.
hip-ep reaches it only through DD's C API (`ops/dd_c_api.h`): five `extern "C"`
functions and plain structs, so no DD C++ types, templates or CRT cross the
boundary. The backend is opt-in, and it is one of the NPU backends described in
[npu-backends.md](npu-backends.md).

## Enabling

| Setting | Where | Effect |
|---|---|---|
| `use_dynamic_dispatch` = `1`/`true` | EP provider option | Lower supported GEMM/Conv shapes to DD |
| `HIPEP_USE_DYNAMIC_DISPATCH` = `1`/`true` | Environment, read at compile time | Same, when the provider option is unset |
| `DYNAMICDISPATCH_ROOT` (or `DD_ROOT`) | Environment or CMake, build and run time | Build: where `ops/dd_c_api.h` is found (`include/ryzenai/dynamic_dispatch` or `include` below it). Run: `bin/dyn_dispatch_core.dll` (or the root itself), else the default library search path |

In `hip-mlir-opt` the option is `use-dynamic-dispatch` of
`--hip-to-llvm-pipeline` and `--convert-hip-to-llvm`; in
`compilation_options.fbs` it is `use_dynamic_dispatch`.

## Building and loading

Only the C API header is needed to build. With it, `lib/Runtime` builds the real
`hipdnn-ep-dd` wrappers (`real/dynamic_dispatch.cpp`); without it, or in mock
builds, a stub (`mock/dynamic_dispatch_mock.cpp`) fails every call. The
configure step prints `-- DynamicDispatch C API: <dir>` or that it was not
found.

`dyn_dispatch_core` is loaded on first use, not linked, so the EP loads on
machines without DynamicDispatch. The five entry points
(`dd_combined_gemm_create`, `dd_iconv_create`, `dd_op_initialize_weights`,
`dd_op_execute`, `dd_op_destroy`) are resolved by name; a missing library or
symbol is reported once as `[DD] ...` and every DD op then fails.
`dyn_bins.dll` and DD's other DLLs are found beside `dyn_dispatch_core.dll`;
XRT normally comes from the NPU driver. The wrappers are native code, like the
CoreLib wrappers, and reach the JIT as absolute symbols (`LlvmIrJit`).

DynamicDispatch's C API is added by its PR VitisAI/DynamicDispatch#5617; build
DD from that branch until it merges.

## Which operations run on the NPU

DD kernels exist only for a few exact shapes, listed in
`include/hip/dynamic_dispatch_support.h`, which the HipToLLVM lowering and the
runtime wrappers share. With the option on, the DD lowering patterns are added
ahead of the GPU patterns and match only operations DD can run; every other
operation lowers to the GPU as usual.

| Operation | Runs on the NPU when |
|---|---|
| `hip.matmul`, `hip.gemm` | f16, rank 2, static shape, no transpose, no bias, alpha 1, and `[1, 1024] x [1024, N]` with N in {192, 384} (combined_gemm `{1, B, 1, 1024, 64}`, B in {3, 6}) |
| `hip.conv` | f16, static shape, N = 1, input and output channels multiples of 8, no dilation, group 1, no bias, no padding |

iconv's supported spatial shapes are only known to DD at run time, so a Conv
that passes these checks can still fail when its DD op is created.

## Limitations

- The xclbin/model names are fixed (`8x4_PSMU_ST1_v3.2` for GEMM, `8x4PSP1` for
  Conv), the QDQ tensors are zero-filled and the f16 weights are reinterpreted
  as uint8, so NPU results are not numerically meaningful yet. DD's
  `tests/test_c_api.cpp` issues the same call sequence; keep the two in sync.
- A failed DD call is logged but has no GPU fallback.
- Every call stages its tensors through host memory.
- Native (`.dll`/`.so`) model artifacts do not resolve the DD wrappers; use the
  default bitcode artifacts.

## Benchmark model

`tools/dd-model/make_dd_matmul_model.py` writes a model made only of supported
MatMuls (a fan-out of `[1, 1024] x [1024, 384]` MatMuls joined by a Concat):

```bash
python tools/dd-model/make_dd_matmul_model.py -o dd_matmul_fanout.onnx
set HIPEP_USE_DYNAMIC_DISPATCH=1
onnxruntime_perf_test -t 20 -c 1 -s -I --plugin_ep_libs "AMDGPUExecutionProvider|<install>\bin\hipgpu.dll" --plugin_eps "AMDGPUExecutionProvider" dd_matmul_fanout.onnx
```

Each MatMul creates one DD op, logged as
`[DD] wrap_dd_matmul: dd_combined_gemm_create`. The runtime logs each op's
first `dd_op_execute`, every failed one, and its `dd_op_destroy`; set
`HIPDNN_EP_DEBUG=1` to log every execute (not for benchmarking).
`HIPEP_DD_DEBUG_NO_GET_OP_STATE=1` recreates the DD op on every call instead
of caching it.

## References

- Wrappers: `lib/Runtime/real/dynamic_dispatch.cpp`; stub:
  `lib/Runtime/mock/dynamic_dispatch_mock.cpp`; build:
  `lib/Runtime/CMakeLists.txt` STEP 6c.
- Lowering: `lib/Conversion/HipToLLVM/DynamicDispatchLowering.cpp`.
- Tests: `test/lit/Conversion/hip-to-llvm/test_dd_gemm.mlir`,
  `test_dd_conv.mlir`, `test/python/test_dd_matmul.py`.
