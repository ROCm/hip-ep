<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper on AMD GPU — Demo & Python Quick Reference

A short, self-contained guide for **running OpenAI Whisper speech-to-text on an
AMD GPU** through the MorphiZen Execution Provider — built for a live technical
demo. For the full build-from-scratch walkthrough see
[whisper_quick_start.md](whisper_quick_start.md).

Whisper runs **fp16 by default** (body fp16, `lm_head` fp32 → greedy decoding is
argmax-lossless and faster than fp32 on the GPU).

---

## TL;DR — one command

```bash
conda activate hipdnn-ep
hf auth login                 # one-time; the AMD Whisper repos are gated
python scripts/whisper_demo.py
```

This downloads `large-v3-turbo` from Hugging Face, fetches the bundled JFK clip,
transcribes it on the GPU, and prints the transcription plus a latency / real-time
factor table. First run downloads the model (~1.6 GB) and compiles the GPU graphs;
subsequent runs are instant.

---

## Supported variants (all auto-download from AMD HF, fp16)

| Variant          | Size    | Hugging Face repo                          |
|------------------|---------|--------------------------------------------|
| `tiny`           | ~75 MB  | `amd/whisper-tiny-onnx-fp16`               |
| `base`           | ~145 MB | `amd/whisper-base-onnx-fp16`               |
| `small`          | ~490 MB | `amd/whisper-small-onnx-fp16`              |
| `medium`         | ~1.5 GB | `amd/whisper-medium-onnx-fp16`             |
| `large-v3-turbo` | ~1.6 GB | `amd/whisper-large-v3-turbo-onnx-fp16`     |
| `large-v3`       | ~3.1 GB | `amd/whisper-large-v3-onnx-fp16` (+ fp32)  |

`tiny` is best for the quickest smoke run; `large-v3-turbo` (the demo default) is
the best accuracy/speed balance for a meeting.

---

## The demo script

```bash
python scripts/whisper_demo.py                      # large-v3-turbo, JFK clip, GPU
python scripts/whisper_demo.py --variant tiny       # fastest / smallest
python scripts/whisper_demo.py --audio meeting.wav  # your own 16 kHz mono clip
python scripts/whisper_demo.py --compare-cpu        # GPU vs ORT CPU (same text)
python scripts/whisper_demo.py --list               # list variants
```

Your audio must be **16 kHz mono, ≤ 30 s**. Convert anything else first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 clip_16k.wav
```

Expected output (abridged):

```
================================================================
Whisper demo — large-v3-turbo (fp16) on the AMD GPU
================================================================
  -> audio: .../jfk.wav  (11.0 s, expects 16 kHz mono)
  -> model ready: .../models/whisper-large-v3-turbo-onnx-fp16
  -> transcribing on the GPU (timed) ...
================================================================
Transcription
================================================================
  ' And so my fellow Americans, ask not what your country can do for you, ...'
================================================================
Performance — AMD GPU (fp16)
================================================================
  audio length      :     11.00 s
  encoder           :     ...   ms
  prefill (TTFT)    :     ...   ms
  decode loop       :     ...   ms   (N tokens)
  total compute     :     ...   ms
  decode throughput :     ...   tok/s
  real-time factor  :     0.xxx   (Nx faster than real time)
================================================================
```

---

## Running Whisper from Python (no script)

The demo and tests both drive the shared harness in
[`test/python/whisper/whisper_infer.py`](../test/python/whisper/whisper_infer.py).
The minimal flow:

```python
import pathlib, sys
import numpy as np

REPO_ROOT = pathlib.Path("/path/to/hip-ep")
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))
sys.path.insert(0, str(REPO_ROOT / "test" / "python" / "whisper"))

import whisper_infer
from conftest import setup_whisper_variant

# 1. Download (from amd/whisper-<variant>-onnx-fp16) + prepare the model.
#    Returns the model dir and a WhisperVariant carrying shapes + start tokens.
model_dir, variant = setup_whisper_variant("large-v3-turbo", precision="fp16")

# 2. Load a 16 kHz mono wav -> log-mel features.
audio_fp = whisper_infer.load_audio_features(
    REPO_ROOT / "test/python/data/whisper/jfk.wav", variant=variant
)

# 3. GPU greedy decode (encoder -> prefill -> decode loop).
gpu = whisper_infer.make_morphizen_session_factory(REPO_ROOT, model_dir)
tokens = whisper_infer.greedy_decode_morphizen(
    gpu, audio_fp, dtype=np.float16, variant=variant
)

# 4. Detokenize.
text = whisper_infer.decode_text(
    tokens, tokenizer_id=variant.hf_model_id, eot=variant.eot
)
print(text)
```

To run the **ORT CPU** reference instead, swap step 3 for
`whisper_infer.make_cpu_session_factory(model_dir)` +
`whisper_infer.greedy_decode_cpu(...)` (uses the dynamic fp32 graphs — the only
valid CPU reference).

---

## Prerequisites (one-time)

1. **Build the EP** — `python build.py` (see whisper_quick_start.md §1). This
   produces the AMDGPU umbrella EP that the session factory loads.
2. **Environment every shell** — `conda activate hipdnn-ep`, and put
   `install/therock/bin` + `install/dist/bin` on `PATH` with `THEROCK_DIST`
   pointing at `install/therock` (so the EP can link the compiled model DLL).
3. **Hugging Face auth** — `hf auth login` (the AMD Whisper repos are gated).

If the model can't be downloaded, the helpers raise a clear error pointing at
both the `hf auth login` remedy and the local-build backup
(`python scripts/build_whisper_models.py --variant <name>`).

---

## Validate it works

```bash
# Per-phase correctness (encoder/prefill/decode cosine) + e2e greedy == JFK quote,
# fp16 across every variant + fp32 on large-v3:
pytest test/python/whisper/test_whisper.py -v -s

# Just one quick variant:
pytest test/python/whisper/test_whisper.py -v -s -k "tiny-fp16"
```
