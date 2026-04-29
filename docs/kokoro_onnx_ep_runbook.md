<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Kokoro ONNX EP Runbook

This branch adds the Kokoro operator coverage needed to run the ONNX model with the MorphiZen EP, plus a conservative standard ONNX Q/DQ path.

## Branch Changes

- Added Kokoro runtime/conversion support for STFT, LSTM, ConvTranspose, Resize, Pad, Expand, NonZero, ScatterND, CumSum, comparisons, reductions, data movement, and elementwise ops.
- Stabilized sensitive FP16 MIOpen paths by running LSTM, Conv, and Softmax internally in FP32 and casting back.
- Added standard ONNX Q/DQ handling through validated cast/elementwise decomposition, constant DQ folding, and smoke coverage for `QLinearMatMul`, `MatMulInteger`, and `QLinearConv`.
- Kept GPU runtime startup behind explicit opt-in because the FP16 investigation previously triggered Windows `LiveKernelEvent 141` watchdog resets.

Use Q/DQ or FP32 for quality-sensitive runs. FP16 EP is finite and non-silent, but its voice quality remains worse because of pitch/source artifacts.

Reference quality snapshot:

```text
CPU_ONNX      dur=3.100s rms=0.080251 hf=0.4017
FP32_EP       dur=3.100s rms=0.083199 hf=0.3665
QDQ_EP_SMOKE  dur=3.100s rms=0.083188 hf=0.3665
GGUF          dur=3.000s rms=0.056262 hf=0.3629
FP16_EP       dur=3.100s rms=0.071007 hf=0.4173
```

## Environment

Run from a Visual Studio x64 developer shell, or after `vcvarsall.bat x64`:

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

## Run FP32 ONNX EP

```cmd
%LEMONDATE_ROOT%\build\bin\kokoro-onnx-server.exe ^
  --model %LEMONDATE_ROOT%\models\kokoro-v1.0.onnx ^
  --ep-dll %EP_BIN%\onnxruntime_morphizen_ep.dll ^
  --gguf %LEMONDATE_ROOT%\models\Kokoro_no_espeak_Q4.gguf ^
  --voices-dir %DEMOS_ROOT%\voices ^
  --port 15002
```

Generate a WAV:

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

## Build and Run Q/DQ

Create the validated smoke model:

```cmd
D:\jam\demos\.venv\Scripts\python.exe ^
  D:\jam\demos\scripts\quantize_kokoro_qdq.py ^
  --dst D:\jam\lemondate\models\kokoro-v1.0-local-qdq-smoke.onnx ^
  --max-weights 16 ^
  --run-cpu
```

Omit `--max-weights` to create `D:\jam\lemondate\models\kokoro-v1.0-local-qdq.onnx`. The full Q/DQ model still exposes a compiler/runtime stress case, so increase `--max-weights` gradually when expanding coverage.

Run the smoke model:

```cmd
%LEMONDATE_ROOT%\build\bin\kokoro-onnx-server.exe ^
  --model %LEMONDATE_ROOT%\models\kokoro-v1.0-local-qdq-smoke.onnx ^
  --ep-dll %EP_BIN%\onnxruntime_morphizen_ep.dll ^
  --gguf %LEMONDATE_ROOT%\models\Kokoro_no_espeak_Q4.gguf ^
  --voices-dir %DEMOS_ROOT%\voices ^
  --port 15002
```

Use the same PowerShell request above and change `-OutFile` to `D:\jam\demos\qdq_ep_test.wav`.

## Validation

```cmd
set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
D:\jam\demos\.venv\Scripts\python.exe D:\jam\demos\scripts\verify_qdq_quant_ops.py
```

Expected output ends with:

```text
All Q/DQ quantized op checks passed.
```

The full quality/performance report is in `D:\jam\demos\FINAL_ONNX_EP_QUALITY_PERF_REPORT.md`.

## Limits

Q/DQ smoke preserves FP32-like quality but is not faster than FP32 EP yet because the stable path still decomposes into float kernels. GGUF remains much faster until the EP has fused int8 Conv/Gemm/MatMul kernels for Q/DQ weight paths.
