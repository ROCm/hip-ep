<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# FP16 remote reproduction plan

This is the safe path for continuing Kokoro FP16 EP debugging without risking
the display-attached Windows workstation.

## Preconditions

- Run on a non-display test machine or a machine where a GPU watchdog reset is
  acceptable.
- Activate the same Python venv pattern used in `D:\jam\demos`.
- Use `rocm-sdk path --root` / `--bin` from that venv. Do not point builds at a
  source checkout of TheRock.
- Set the runtime opt-in only in the remote test shell:

```cmd
set HIPDNN_EP_ALLOW_GPU_RUNTIME=1
set HIPDNN_EP_ENABLE_GRAPHS=
set MORPHIZEN_NO_BUFFER_OPT=1
set HIPDNN_EP_SYNC_OPS=1
set HIPDNN_EP_AUTOTUNE=0
set HIP_CUSTOM_KERNELS_DIR=D:\jam\prebuilt-local\lib
```

## Repro order

Start with isolated ops, not full Kokoro:

0. FP16 Kokoro session compilation only.
   - Current working path: start `kokoro-onnx-server` and stop after `ready`,
     before sending `/v1/audio/speech`.
   - Avoid `D:\jam\demos\scripts\compile_only_fp16_session.py` for now; Python
     ORT registration does not expose `MorphiZenExecutionProvider` to
     `InferenceSession` in this environment.
1. FP16 `Sigmoid`
2. FP16 `ReduceSum`
3. FP16 `Cast(f16 -> i64)`
4. Combined duration chain:
   `Sigmoid -> ReduceSum -> Div -> Round -> Clip -> Cast -> CumSum`
5. Full `kokoro-v1.0-local-fp16.onnx`

Stop at the first hang or mismatch.

Latest local result: compile/session creation reached `ready`, but a single
guarded full Kokoro FP16 inference request still produced the broken 43.75 s
waveform and fresh watchdog reports. Continue inference debugging only on the
safe remote/non-display target.

`scripts/run_fp16_repro.ps1` prepares the remote shell environment and verifies
the explicit opt-in, but intentionally does not launch `hip-onnx-runner` itself.
Use it as a setup step, then run one isolated test at a time.

## Expected references from the display workstation

Known-good quality baseline for FP32/GGUF:

```text
CPU_ONNX dur=3.100s rms=0.080242 diff_rms=0.032231 hf=0.4017
EP_ONNX  dur=3.100s rms=0.083199 diff_rms=0.030492 hf=0.3665
GGUF     dur=3.000s rms=0.056199 diff_rms=0.020468 hf=0.3642
```

Current FP16 split:

```text
FP16_CPU dur=3.100s raw=3.875s rms=0.063739 diff_rms=0.032930 hf=0.5166
FP16_EP  dur=21.900s raw=43.750s rms=0.823046 diff_rms=0.797948 hf=0.9695
```

This means the converted FP16 model is structurally sane under CPU ORT, but the
EP/runtime path corrupts or miscomputes the FP16 graph.

## Current leading suspicion

The first tight repro that hung locally was isolated FP16 `Sigmoid`.
Duration prediction depends on Sigmoid, and the broken full model emits a much
larger/noisy waveform. The in-progress source change routes ONNX Sigmoid through
the custom HIP unary kernel instead of the MIOpen activation path.

Do not validate that change on the display workstation.
