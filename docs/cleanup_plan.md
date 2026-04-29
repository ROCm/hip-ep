<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Cleanup plan

This is the cleanup checklist after guarded local FP16 runtime validation.

## Keep immediately

These changes are safety-only or already validated by source inspection:

- `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
  - requires `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` before loading generated runtime
    DLLs.
- `tools/hip-onnx-runner/hip-onnx-runner.cpp`
  - requires `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` before registering the EP.
- `lib/CInterface/CompilerAPI.cpp`
  - requires `HIPDNN_EP_ENABLE_GRAPHS=1` before compile-time hipDNN graph
    compilation.
- `docs/gpu_watchdog_safety.md`
- `docs/fp16_remote_repro.md`
- `docs/current_upstream_delta.md`

## Validated FP16 fixes to keep

These FP16 fixes were validated by the guarded three-way quality check on
2026-04-29:

```text
CPU_ONNX dur=3.100s rms=0.080251 diff_rms=0.032239 hf=0.4017
EP_ONNX  dur=3.100s rms=0.071007 diff_rms=0.029629 hf=0.4173
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

No new WER/watchdog event appeared after that run.

- `3rd-party/custom_kernels/hip/cast_kernel.hip`
- `lib/Runtime/real/cast.cpp`
- `3rd-party/custom_kernels/hip/reduce_sum_kernel.hip`
- `lib/Runtime/real/reduce_sum.cpp`
- `3rd-party/custom_kernels/hip/reduce_mean_kernel.hip`
- `lib/Runtime/real/reduce_mean.cpp`
- `3rd-party/custom_kernels/hip/slice_kernel.hip`
- `3rd-party/custom_kernels/hip/transpose_kernel.hip`
- `lib/Runtime/real/nonzero.cpp`
- `3rd-party/custom_kernels/include/hip_custom_kernels.h`
- `3rd-party/custom_kernels/hip/elementwise_unary_kernel.hip`
- `lib/Conversion/OnnxToHip/ActivationConversion.cpp`
- `lib/Runtime/real/lstm.cpp`
- `lib/Runtime/real/miopen.cpp`
- `lib/Runtime/real/elementwise.cpp`
- `lib/Runtime/real/elementwise_unary.cpp`

The last source-generator fix keeps Kokoro's phase tensor unscaled through the
linear resize and applies the `300x` scale inside the following 9-harmonic
`Sin` kernel in FP32, avoiding FP16 phase overflow.

Temporary runtime `DURATION_TRACE` helpers used to isolate NaNs were removed
after validation; keep future diagnostics behind the existing `nan_trace`
mechanism or `RUNTIME_DEBUG_LOG`.

## Keep if already part of FP32-quality fix

These files include changes that produced the FP32 EP vs GGUF quality match.
Separate those from FP16-only experiments during final cleanup:

- `3rd-party/custom_kernels/hip/stft_kernel.hip`
- `lib/Runtime/real/stft.cpp`
- `lib/Conversion/HipToLLVM/StftLowering.cpp`
- `lib/Conversion/OnnxToHip/StftConversion.cpp`
- `lib/Conversion/OnnxToHip/ConvConversion.cpp`
- `lib/Conversion/OnnxToHip/OnnxToHip.cpp`
- `lib/Conversion/OnnxToHip/OnnxToHipUtils.h`
- `lib/Conversion/OnnxToHip/ReduceSumConversion.cpp`
- `lib/Conversion/OnnxToHip/Tier2ShapeConversion.cpp`
- `lib/Conversion/OnnxToHip/Tier6Conversion.cpp`
- `lib/Conversion/OnnxToHip/TransposeConversion.cpp`

Known validated FP32 quality metrics:

```text
CPU_ONNX dur=3.100s rms=0.080242 diff_rms=0.032231 hf=0.4017
EP_ONNX  dur=3.100s rms=0.083199 diff_rms=0.030492 hf=0.3665
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

## Do not keep in final commit unless still useful

These are debugging helpers and should not be committed by default:

- temporary generated `.onnx` probe models
- `_i_dump` / `_o_dump` directories
- intermediate `.bin` dumps
- one-off Python probes in `D:\jam\demos\scripts` unless they become a real
  regression test

If a probe becomes a test, make it:

- guarded by `HIPDNN_EP_ALLOW_GPU_RUNTIME=1`
- deterministic
- runnable on a safe CI/test machine
- documented as private EP validation, not public lemondate workflow

The public demos workspace has a local scanner:

```powershell
D:\jam\demos\.venv\Scripts\python.exe D:\jam\demos\scripts\scan_onnx_ep_safety.py
```

As of the latest checkpoint, the scanner reports:

```text
risky scripts: 34
unguarded:     0
```

Keep new EP-touching helper scripts behind the same
`HIPDNN_EP_ALLOW_GPU_RUNTIME=1` guard.

## Public repo caution

Do not publish private EP implementation details into public repositories. Public
repos may expose a generic ONNX-EP backend hook, but must not hardcode private
repo paths or MorphiZen/hipDNN internals unless the project owner explicitly
decides to publish that work.
