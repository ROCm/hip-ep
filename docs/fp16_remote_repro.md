<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Remote reproduction plan

This is the safer path for risky Kokoro EP experiments that may stress the GPU
driver.  The display-attached Windows workstation can run guarded single-shot
validation, but long or exploratory EP runs should still use a remote or
non-display test machine.

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

0. Kokoro session compilation only.
   - Start `kokoro-onnx-server` and stop after `ready`, before sending
     `/v1/audio/speech`.
1. FP16 `Sigmoid`
2. FP16 `ReduceSum`
3. FP16 `Cast(f16 -> i64)`
4. Combined duration chain:
   `Sigmoid -> ReduceSum -> Div -> Round -> Clip -> Cast -> CumSum`
5. Full `kokoro-v1.0-local-fp16.onnx`
6. Q/DQ isolated validation via `D:\jam\demos\scripts\verify_qdq_quant_ops.py`
7. Q/DQ smoke/full Kokoro variants from
   `D:\jam\demos\scripts\quantize_kokoro_qdq.py`

Stop at the first hang or mismatch.

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

Current FP16 status:

```text
FP16_EP dur=3.100s rms=0.071007 diff_rms=0.029629 hf=0.4173
```

The current FP16 EP path is finite and non-silent, but subjective voice quality
is still worse than FP32/GGUF due to pitch/source artifacts. Prefer Q/DQ for
quality-preserving quantization work.

Current Q/DQ smoke status:

```text
QDQ_EP_SMOKE dur=3.100s rms=0.083188 hf=0.3665
QDQ_CPU      dur=3.100s rms=0.080239 hf=0.4017
```

The full Q/DQ model still exposes a compiler/runtime limitation; expand the
smoke policy gradually and validate after each increase.
