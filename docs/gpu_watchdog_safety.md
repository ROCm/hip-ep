<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GPU watchdog safety notes

## Status

The Kokoro FP16 EP path previously produced Windows Error Reporting
`LiveKernelEvent` reports with `P1: 141` and `WATCHDOG-*.dmp` files under
`C:\WINDOWS\LiveKernelReports\WATCHDOG`. The runtime is still guarded because
it initializes AMD GPU libraries and can stress the display driver, but the
2026-04-29 guarded FP16 validation completed without a new WER/watchdog event.

Observed reports from 2026-04-29 include:

- `WATCHDOG-20260429-0413.dmp`
- `WATCHDOG-20260429-0416.dmp`
- `WATCHDOG-20260429-0445.dmp`
- `WATCHDOG-20260429-0504.dmp`
- `WATCHDOG-20260429-0511.dmp`
- `WATCHDOG-20260429-0512.dmp`
- `WATCHDOG-20260429-0518.dmp`
- `WATCHDOG-20260429-0816.dmp`
- `WATCHDOG-20260429-0817.dmp`
- `WATCHDOG-20260429-0819.dmp`

The later `Kernel-Power 41` events are manual power-off aftermath, not root
cause evidence. The useful signal is `LiveKernelEvent 141`, which indicates a
GPU/display-driver watchdog condition.

## Unsafe paths

Do not run these on the display workstation unless intentionally opting in:

- `hip-onnx-runner.exe` without `-n`
- `kokoro-onnx-server.exe`
- any `run_onnx_server*.bat`
- old scratch/debug Python scripts that spawn `hip-onnx-runner.exe`
- any ORT session that registers `onnxruntime_morphizen_ep.dll`
- any generated model DLL `LoadLibrary` / `inference_init` path

The dangerous runtime sequence is:

1. ORT creates a session and asks the EP for supported nodes.
2. The EP compiles the fused graph into a generated model DLL.
3. `InferenceState::create()` writes that DLL to a temp path and loads it.
4. `inference_init()` initializes GPU runtime state:
   - `hipGetDeviceCount`
   - `hipSetDevice`
   - `hipGetDeviceProperties`
   - `hipStreamCreate`
   - `miopenCreate`
   - `hipblasLtCreate`

This path can trigger the watchdog even for tiny models such as an isolated
FP16 `Sigmoid`.

## Safety guards

The runtime now requires:

```cmd
set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
```

before loading generated EP runtime DLLs. This guard is intentionally separate
from `HIPDNN_EP_ENABLE_GRAPHS`; graph compilation can be disabled while the
plain generated-runtime path is still unsafe.

Guarded source locations:

- `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
- `tools/hip-onnx-runner/hip-onnx-runner.cpp`

Lemondate's ONNX server has the same guard locally:

- `D:\jam\lemondate\src\kokoro-onnx-server\main.cpp`

The lemondate orchestrator also refuses to launch the ONNX backend unless the
same opt-in is set:

- `D:\jam\lemondate\src\lemond\src\cpp\server\backends\kokoro_server.cpp`

## Safe continuation plan

Prefer remote/non-display machines for risky EP runtime investigation:

1. Use a remote/non-display machine for any EP runtime/model execution.
2. Set `HIPDNN_EP_ALLOW_GPU_RUNTIME=1` only in that remote shell.
3. Start with the smallest repro:
   - FP16 `Sigmoid`
   - FP16 `ReduceSum`
   - FP16 `Cast(f16 -> i64)`
4. Only after those pass, rerun Kokoro FP16 endpoint quality.

On the display workstation, prefer:

- source inspection
- CPU ORT reference runs
- documentation
- code cleanup
- non-GPU builds if needed
- guarded single-shot EP validation only when `HIPDNN_EP_ALLOW_GPU_RUNTIME=1`
  is explicitly set and the machine owner accepts the risk

2026-04-29 historical note: before the FP16 runtime fixes landed, guarded
full-model inference produced invalid full-buffer audio and fresh
`LiveKernelEvent 141` reports.  The first Sigmoid-route rebuild still produced
the same 43.75 s waveform and `WATCHDOG-20260429-0819.dmp`.

2026-04-29 final update: after FP16 LSTM/Conv/Softmax upcasting and the Kokoro
source-generator phase overflow fix, the guarded three-way quality check
completed successfully:

```text
CPU_ONNX dur=3.100s rms=0.080251 diff_rms=0.032239 hf=0.4017
EP_ONNX  dur=3.100s rms=0.071007 diff_rms=0.029629 hf=0.4173
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

Recent Application/System event checks after the validation run did not show a
new WER, `LiveKernelEvent`, or display-driver watchdog entry.

Q/DQ smoke validation also completed successfully and preserves FP32-like
quality.  The full Q/DQ model is still a compiler/runtime stress case, so keep
full-model Q/DQ iteration remote or single-shot guarded.
