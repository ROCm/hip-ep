<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Current upstream delta

This document records the current uncommitted delta for Kokoro FP16 EP work.

## Verified FP32 quality baseline

The existing three-way quality check uses:

- CPU ONNX reference
- FP32 ONNX EP server
- GGUF HIP server

Last reproduced metrics:

```text
CPU_ONNX dur=3.100s rms=0.080242 diff_rms=0.032231 hf=0.4017
EP_ONNX  dur=3.100s rms=0.083199 diff_rms=0.030492 hf=0.3665
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

Interpretation: FP32 EP quality is consistent with GGUF.

## Current FP16 status

2026-04-29 final local guarded validation: the FP16 EP path now produces
finite, non-silent audio and matches the FP32 CPU duration. No new WER or
System watchdog event appeared in the 20-minute window after validation.

```text
CPU_ONNX dur=3.100s rms=0.080251 diff_rms=0.032239 hf=0.4017
EP_ONNX  dur=3.100s rms=0.071007 diff_rms=0.029629 hf=0.4173
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

The final FP16-specific fixes that changed the bad-output behavior were:

- FP16 LSTM internally upcasts inputs/weights/states to FP32 for MIOpen RNN and
  casts outputs back to FP16.
- FP16 Conv internally upcasts inputs/weights/bias to FP32 for MIOpen Conv and
  casts outputs back to FP16.
- FP16 Softmax internally upcasts to FP32 for MIOpen Softmax and casts output
  back to FP16.
- Kokoro's source-generator phase path avoids the pre-`Resize` `*300` FP16
  overflow by carrying the unscaled phase through the linear resize, then
  applying the `300x` scale inside the following 9-harmonic `Sin` kernel in
  FP32.

The guarded raw-output probe now reports:

```text
n=2097152 nonzero=83523 nan=0 inf=0
range=[-0.520996,0.555664] mean_abs=0.00146713
```

Historical notes follow.

CPU ORT on `kokoro-v1.0-local-fp16.onnx` is structurally sane:

```text
FP16_CPU dur=3.100s raw=3.875s rms=0.063739 diff_rms=0.032930 hf=0.5166
```

EP runtime output is broken:

```text
FP16_EP dur=21.900s raw=43.750s rms=0.823046 diff_rms=0.797948 hf=0.9695
```

The failure is EP/runtime-specific, not simply a bad converted model.

For compile-only validation, use:

- `D:\jam\demos\scripts\compile_only_fp16_session.py`

The pure Python variant is currently not viable with this ORT binding:
`register_execution_provider_library()` returns without making
`MorphiZenExecutionProvider` available to Python `InferenceSession`, so ORT
falls back to CPU and errors because CPU fallback is disabled.

The C++ `kokoro-onnx-server` compile/session path was tested with
`HIPDNN_EP_ALLOW_GPU_RUNTIME=1` and no inference request:

```text
[get_supported_nodes] Found 2503 supported (of 2503 meta_def)
[SHARED_CONSTANTS] Published new blob (162317906 bytes)
[kokoro-onnx] session created in 7313 ms
[kokoro-onnx] ready (sr=24000 Hz, default_voice=af_heart)
```

No new WER/watchdog event appeared after that compile-only run.

Guarded inference was then tested once:

```text
HTTP 200, 2100044 bytes
FP16_EP_GUARDED dur=21.900s raw=43.750s rms=0.823046 diff_rms=0.797948 hf=0.9695 peak=0.999969
```

That inference run produced fresh `LiveKernelEvent 141` watchdog reports:

```text
C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-20260429-0816.dmp
C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-20260429-0817.dmp
```

Conclusion: compile/session creation is stable enough to reach `ready`, but the
FP16 inference path still both produces invalid audio and trips the GPU
watchdog. Do not continue local inference testing on the display workstation.

After rebuilding the candidate Sigmoid custom-kernel route, the same single
inference was repeated:

```text
FP16_EP_SIGMOID_FIX dur=21.900s raw=43.750s rms=0.823046 diff_rms=0.797948 hf=0.9695 peak=0.999969
```

It produced another watchdog report:

```text
C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-20260429-0819.dmp
```

Conclusion: routing ONNX `Sigmoid` away from the MIOpen activation path is not
the full-model fix. The next likely culprit is FP16 LSTM / downstream MIOpen
runtime, because full-model inference enters `wrap_miopenRNNForwardInference`
with `dtype=f16` before the bad output.

## Safety / watchdog changes

These are intended to remain:

- `lib/CInterface/CompilerAPI.cpp`
  - `HIPDNN_EP_ENABLE_GRAPHS=1` is now required before compile-time hipDNN
    graph compilation is enabled.
- `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
  - `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` is now required before loading generated
    runtime DLLs and calling `inference_init`.
- `tools/hip-onnx-runner/hip-onnx-runner.cpp`
  - `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` is now required before registering the EP.

Local lemondate mirror:

- `D:\jam\lemondate\src\kokoro-onnx-server\main.cpp`
  - same `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` guard before registering/using the EP.
- `D:\jam\lemondate\src\lemond\src\cpp\server\backends\kokoro_server.cpp`
  - same guard before launching `kokoro-onnx-server`.

Local demo launcher guards:

- `D:\jam\demos\run_onnx_server_guard.bat`
- every `D:\jam\demos\run_onnx_server*.bat` now calls that guard before
  launching `kokoro-onnx-server.exe`.

These guards have not been built into binaries on the display workstation. They
are source-level and script-level protection only.

## FP16 runtime fix notes

These changes are part of the validated FP16 path:

- `3rd-party/custom_kernels/hip/cast_kernel.hip`
  - adds GPU-only FP16 cast paths, including `int64 <-> fp16` and
    `fp16 <-> int8/uint8`.
- `lib/Runtime/real/cast.cpp`
  - removes CPU host fallback after failed GPU cast.
- `3rd-party/custom_kernels/hip/reduce_sum_kernel.hip`
  - adds FP16 `ReduceSum` with FP32 accumulation and FP16 output.
- `lib/Runtime/real/reduce_sum.cpp`
  - checks outputs using the correct element size.
- `3rd-party/custom_kernels/hip/reduce_mean_kernel.hip`
  - adds explicit FP16 `ReduceMean` kernels and clears stale HIP launch errors.
- `lib/Runtime/real/reduce_mean.cpp`
  - checks outputs using the correct element size.
- `3rd-party/custom_kernels/hip/slice_kernel.hip`
  - clears stale HIP launch error state before launching `Slice`.
- `3rd-party/custom_kernels/hip/transpose_kernel.hip`
  - clears stale HIP launch error state before launching `Transpose`.
- `lib/Runtime/real/nonzero.cpp`
  - treats `k_max == 0` / empty input as a successful empty output.
- `3rd-party/custom_kernels/include/hip_custom_kernels.h`
  - adds `HIP_UNARY_SIGMOID`.
- `3rd-party/custom_kernels/hip/elementwise_unary_kernel.hip`
  - implements `HIP_UNARY_SIGMOID`.
- `lib/Conversion/OnnxToHip/ActivationConversion.cpp`
  - routes ONNX `Sigmoid` through `hip.unary_elementwise(kind=SIGMOID)`
    instead of the MIOpen activation path.

Temporary `HIPDNN_EP_TRACE_DURATION` / `[DURATION_TRACE]` helpers used during
NaN isolation were removed after the source-generator overflow fix was
validated.

## FP32 quality fixes already worth preserving

These changes predate the watchdog pause and produced the FP32 EP quality match
against GGUF:

- STFT phase canonicalization / STFT runtime updates.
- Attribute handling for DenseI64ArrayAttr-style ONNX attributes.
- Conversion improvements for dynamic shapes and output sizing.

Keep these separate from the unverified FP16-specific changes during cleanup.

## Local runtime caution

The EP runtime still initializes AMD GPU libraries and should stay opt-in on
display-attached Windows machines. Run only after explicitly setting
`HIPDNN_EP_ALLOW_GPU_RUNTIME=1`, and prefer single-shot validation unless a
non-display test host is available.

- `hip-onnx-runner.exe` with the EP enabled.
- `kokoro-onnx-server.exe`.
- any `run_onnx_server*.bat` with `HIPDNN_EP_ALLOW_GPU_RUNTIME=1`.
- any FP16 EP isolated op test.

Use `docs/fp16_remote_repro.md` for remote/off-workstation validation when
doing risky iteration.
