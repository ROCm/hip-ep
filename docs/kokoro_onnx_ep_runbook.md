<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Kokoro ONNX EP Runbook

This document summarizes the Kokoro-related EP changes on this branch and how to
run the validated FP32 and standard ONNX Q/DQ variants.

## What Changed

### Safety Guards

The ONNX EP path initializes AMD GPU runtime libraries and has previously
triggered Windows `LiveKernelEvent 141` watchdog resets while the FP16 path was
being debugged.  GPU runtime use is therefore explicit opt-in:

```cmd
set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
```

Guarded paths:

- `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
- `tools/hip-onnx-runner/hip-onnx-runner.cpp`
- `D:\jam\lemondate\src\kokoro-onnx-server\main.cpp`
- `D:\jam\lemondate\src\lemond\src\cpp\server\backends\kokoro_server.cpp`

Keep `HIPDNN_EP_ENABLE_GRAPHS` unset unless intentionally testing hipDNN graph
compilation:

```cmd
set HIPDNN_EP_ENABLE_GRAPHS=
```

### Runtime and Conversion Support

The branch adds the missing Kokoro operator coverage needed for GPU-only ONNX EP
execution, including:

- STFT / iSTFT-related support through rocFFT-backed STFT handling.
- Dynamic-shape conversion and lowering improvements.
- Custom HIP kernels for data movement, reductions, comparisons, padding,
  resize, nonzero, scatter, cumsum, and elementwise ops.
- FP16 stabilization for LSTM, Conv, and Softmax by running sensitive MIOpen
  paths internally in FP32 and casting results back.
- Standard ONNX Q/DQ support:
  - runtime Q/DQ decomposes into validated cast and elementwise paths;
  - constant weight `DequantizeLinear` folds to float constants;
  - `QLinearMatMul`, `MatMulInteger`, and a smoke path for `QLinearConv`
    decompose through the validated Q/DQ and float kernels.

### Quality Status

Validated reference metrics for the current branch:

```text
CPU_ONNX      dur=3.100s rms=0.080251 hf=0.4017
FP32_EP       dur=3.100s rms=0.083199 hf=0.3665
QDQ_EP_SMOKE  dur=3.100s rms=0.083188 hf=0.3665
GGUF          dur=3.000s rms=0.056262 hf=0.3629
FP16_EP       dur=3.100s rms=0.071007 hf=0.4173
```

FP16 EP is finite and non-silent, but subjective voice quality is worse than
FP32/QDQ/GGUF due to pitch/source artifacts.  Use Q/DQ rather than all-FP16 for
quality-preserving quantization work.

## Common Environment

Run these from a Visual Studio x64 developer shell or after `vcvarsall.bat x64`:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set DEMOS_ROOT=D:\jam\demos
set EP_BIN=D:\jam\prebuilt-local\bin
set EP_LIB=D:\jam\prebuilt-local\lib
set LEMONDATE_ROOT=D:\jam\lemondate
set THEROCK_DIST=D:\jam\demos\.venv\Lib\site-packages\_rocm_sdk_devel

set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
set HIPDNN_EP_ENABLE_GRAPHS=
set MORPHIZEN_NO_BUFFER_OPT=1
set HIP_CUSTOM_KERNELS_DIR=%EP_LIB%
set HIPDNN_EP_SYNC_OPS=1
set HIPDNN_EP_AUTOTUNE=0
set PATH=%THEROCK_DIST%\bin;%THEROCK_DIST%\lib;%EP_BIN%;%EP_LIB%;%PATH%
```

## Run Kokoro FP32 ONNX EP

Start the server:

```cmd
%LEMONDATE_ROOT%\build\bin\kokoro-onnx-server.exe ^
  --model %LEMONDATE_ROOT%\models\kokoro-v1.0.onnx ^
  --ep-dll %EP_BIN%\onnxruntime_morphizen_ep.dll ^
  --gguf %LEMONDATE_ROOT%\models\Kokoro_no_espeak_Q4.gguf ^
  --voices-dir %DEMOS_ROOT%\voices ^
  --port 15002
```

Generate a test WAV:

```powershell
$body = @{
  input = "Hello world, this is a test of the text to speech system."
  voice = "af_heart"
  response_format = "wav"
  speed = 1.0
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri "http://127.0.0.1:15002/v1/audio/speech" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" `
  -OutFile "D:\jam\demos\fp32_ep_test.wav"
```

Stop the server when done.

## Build Kokoro Q/DQ Model

The Q/DQ model generator lives in the demos workspace because it consumes local
voice/test inputs:

```cmd
D:\jam\demos\.venv\Scripts\python.exe ^
  D:\jam\demos\scripts\quantize_kokoro_qdq.py ^
  --dst D:\jam\lemondate\models\kokoro-v1.0-local-qdq-smoke.onnx ^
  --max-weights 16 ^
  --run-cpu
```

Notes:

- `--max-weights 16` creates the validated smoke model.
- Omitting `--max-weights` creates the full weight-only Q/DQ model:
  `D:\jam\lemondate\models\kokoro-v1.0-local-qdq.onnx`.
- The full Q/DQ model currently exposes a compiler/runtime stress case.  Expand
  the smoke model gradually and validate after each increase.

## Run Kokoro Q/DQ ONNX EP

Start the server with the Q/DQ smoke model:

```cmd
%LEMONDATE_ROOT%\build\bin\kokoro-onnx-server.exe ^
  --model %LEMONDATE_ROOT%\models\kokoro-v1.0-local-qdq-smoke.onnx ^
  --ep-dll %EP_BIN%\onnxruntime_morphizen_ep.dll ^
  --gguf %LEMONDATE_ROOT%\models\Kokoro_no_espeak_Q4.gguf ^
  --voices-dir %DEMOS_ROOT%\voices ^
  --port 15002
```

Generate a test WAV using the same PowerShell request shown in the FP32 section,
for example:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:15002/v1/audio/speech" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" `
  -OutFile "D:\jam\demos\qdq_ep_test.wav"
```

## Validation Commands

Standard Q/DQ and quantized-op smoke tests:

```cmd
set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
D:\jam\demos\.venv\Scripts\python.exe D:\jam\demos\scripts\verify_qdq_quant_ops.py
```

Expected result:

```text
dq_per_tensor: ok
q_per_axis: ok
qlinear_matmul: ok
matmul_integer: ok
qlinear_conv: ok
All Q/DQ quantized op checks passed.
```

Quality/performance report:

```text
D:\jam\demos\FINAL_ONNX_EP_QUALITY_PERF_REPORT.md
```

## Known Limitations

- Q/DQ smoke preserves FP32-like quality but is not faster than FP32 EP yet.
  The current stable path uses float kernels after Q/DQ decomposition.
- The full Q/DQ Kokoro model still stresses the compiler/runtime.  Use the
  smoke model or gradually increase `--max-weights`.
- GGUF remains much faster until the EP gains fused int8 Conv/Gemm/MatMul
  kernels for Q/DQ weight paths.
