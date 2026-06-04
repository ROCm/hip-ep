<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper-large-v3 Quick Start

End-to-end guide for running **Whisper-large-v3** speech-to-text on the MorphiZen
EP from a fresh checkout: build the EP, download + compile the model, run the
tests, transcribe your own audio, and compare against CPU / Vulkan.

Whisper is **fp32-only** on this stack (the model ships natively as fp32; the
fair cross-backend benchmark is fp32-vs-fp32). The runtime still supports fp16
for the Llama / gpt-oss family — Whisper just doesn't use it.

> **Shell:** commands in this guide are written for **Git Bash** (the repo's
> convention, same as the main [Quick Start](quick_start.md) — launch it from an
> "x64 Native Tools Command Prompt for VS"). The only commands that differ by
> shell are the environment exports in §2, which give a PowerShell equivalent.
> `pytest` / `git` / `python` commands work the same in either shell.

---

## Pipeline overview

Whisper runs as **three separate ONNX graphs**, each compiled to its own GPU
model DLL on first session init. The encoder runs once; the decoder runs as a
prefill (the forced-start tokens in one shot) followed by a single-token decode
loop until end-of-text:

```
jfk.wav ──► feature extraction ──► ENCODER ──► cross-KV
                                                   │
            START_TOKENS ──► DECODER-PREFILL (S=4) ─┤──► first token
                                                   │
                       ┌────► DECODER-DECODE (S=1) ─┤──► next token ──┐
                       └──────────────(loop until <|endoftext|>)──────┘
```

- **feature extraction** (CPU): 16 kHz PCM → log-mel `[1, 128, 3000]` (padded /
  truncated to Whisper's 30 s window).
- **ENCODER** (`encoder_fixed.onnx`): conv1d front-end + 32 transformer layers →
  `hidden_states [1, 1500, 1280]` **and** the cross-attention KV (32 layers ×
  key/value) — computed once, constant for the whole generation.
- **DECODER-PREFILL** (`decoder_fixed_prefill.onnx`, S=4): the 4 forced-start
  tokens (`<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`) in one
  shot → fills self-KV slots [0..3] → first generated token (the TTFT compute).
- **DECODER-DECODE** (`decoder_fixed_decode.onnx`, S=1): one token per step —
  reads the running self-KV + the fixed cross-KV, appends one self-KV slot,
  argmax → next token, until `<|endoftext|>` or the 448-slot self-KV cap.

See §5 (`--metrics`) for the per-phase latency breakdown and RTF.

---

## 0. Prerequisites

Same toolchain as the main [Quick Start](quick_start.md) (MSVC 2022, Ninja,
Python 3, sccache, `gh`). Plus the conda environment, which provides the Python
deps the Whisper tests need (`onnxruntime`, `transformers`, `soundfile`, etc.):

```bash
conda env create -f environment.yml     # one-time
conda activate hipdnn-ep
```

Whisper needs **no extra pip installs** — the test audio is fetched from public
URLs at runtime (no `datasets` / `jiwer` dependency).

---

## 1. Build the EP (one-time, ~minutes with prebuilt deps)

```bash
python build.py
```

This downloads the prebuilt LLVM/MLIR/Protobuf/FlatBuffers + TheRock ROCm SDK,
detects your GPU, and builds the compiler + MorphiZen EP into `install/`. See the
main [Quick Start](quick_start.md) for details and troubleshooting.

After it finishes you should have `install/dist/bin/onnxruntime_morphizen_ep.dll`.

---

## 2. Set up the runtime environment (every shell)

The EP **compiles and links the model into a GPU DLL at session init**, so the
ROCm runtime libraries must be on `PATH` and `THEROCK_DIST` must point at the SDK.
Run this in every shell before any Whisper command.

**Git Bash** (the convention for the rest of this guide):

```bash
cd <repo-root>
conda activate hipdnn-ep
export THEROCK_DIST="$(pwd)/install/therock"
export PATH="$(pwd)/install/therock/bin:$(pwd)/install/dist/bin:$PATH"
```

**PowerShell** (equivalent — note `$env:` syntax and `;`-separated `Path`):

```powershell
cd <repo-root>
conda activate hipdnn-ep
$env:THEROCK_DIST = "$PWD\install\therock"
$env:Path = "$PWD\install\therock\bin;$PWD\install\dist\bin;$env:Path"
```

> **Why this matters:** without `THEROCK_DIST` / `PATH`, the EP fails to link the
> model DLL (`amdhip64_7.dll missing` / `Failed to link DLL`) and silently falls
> back to CPU. If a "GPU" run is suspiciously slow or wrong, check this first.

---

## 3. Download + compile the model

Everything is automated by `setup_whisper_model_dir` — it downloads the ~6 GB
fp32 ONNX from HuggingFace, applies the required ONNX surgery
(`past_sequence_length` input + position-embed / token-embed fixes for the static
shared-buffer KV cache), and runs `fix_shapes` to lock the static shapes. It is
idempotent; first run is dominated by the download (~5–10 min), later runs are
instant.

Run the setup script (works identically in PowerShell and Git Bash):

```
python scripts/setup_whisper_model.py
```

Verify the variants were produced (`ls` on Git Bash, `dir` on PowerShell):

```
models/whisper-large-v3-onnx/
  encoder.onnx (+.data), decoder.onnx (+.data), tokenizer.json, genai_config.json   (originals)
  encoder_fixed.onnx, decoder_surgery.onnx,
  decoder_fixed_prefill.onnx (S=4), decoder_fixed_decode.onnx (S=1)                  (compiled-shape variants)
```

> **"Compiling the model"** happens the first time the EP loads one of the
> `*_fixed*.onnx` files (MorphiZen lowers ONNX → HIP → a GPU `model.dll`, cached
> in `%TEMP%/morphizen_mlir_*`). The pytest runs in the next step trigger it. To
> force a fresh compile, delete the cache — **PowerShell:**
> `Remove-Item "$env:TEMP\morphizen_mlir_*"` · **Git Bash:** `rm -f "$TEMP"/morphizen_mlir_*`.
> **Always do this after pulling a runtime/kernel change** — the cache key is the
> ONNX hash, not the runtime version, so stale DLLs are otherwise reused silently.

---

## 4. Run the tests

The test audio (jfk.wav from whisper.cpp, LibriSpeech clips from the HF
datasets-server) auto-downloads on first run; tests `skip` cleanly if the network
is unreachable.

### 4a. End-to-end transcription (the headline)

Runs the full pipeline on GPU and asserts the transcription matches the ORT CPU
fp32 reference **verbatim**:

Clear the compiled-model cache to force a real compile, then run the test:

```
# PowerShell:  Remove-Item "$env:TEMP\morphizen_mlir_*" -ErrorAction Ignore
# Git Bash:    rm -f "$TEMP"/morphizen_mlir_*
pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -v -s
```

Expected: both the GPU and CPU runs print
*"And so my fellow Americans, ask not what your country can do for you, ask what
you can do for your country."* and the test **passes**. (First run is slow — it
pays the MLIR compile of the encoder + both decoder variants.)

### 4b. Multi-clip correctness + accuracy (WER)

5 LibriSpeech clips + a long concatenated clip. Each asserts **GPU == CPU
verbatim** *and* **WER vs ground truth** within threshold:

```bash
pytest test/python/whisper/test_whisper.py -k "librispeech or long_30s" -v -s
```

### 4c. Per-step correctness

(Single line — runs the same in PowerShell and Git Bash. Use `-k` to select the
three correctness tests.)

```
pytest test/python/whisper/test_whisper.py -k "encoder_correctness or decoder_prefill_correctness or decoder_decode_correctness" -v -s
```

### 4d. Per-op numeric tests (vs ORT CPU)

Conv1d, attention, and LayerNorm, both fp16 and fp32 (one line):

```
pytest test/numeric/tests/test_whisper_encoder_attention.py test/numeric/tests/test_whisper_cross_attention.py test/numeric/tests/test_whisper_self_attention.py test/numeric/tests/test_conv1d.py test/numeric/tests/test_layer_norm.py --backend ort_ep --ep-name MorphiZenExecutionProvider --ep-dll install/dist/bin/onnxruntime_morphizen_ep.dll --ep-option config_file=install/dist/bin/morphizen_config.json -v
```

### 4e. MLIR conversion (LIT)

```
ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests
```

### Run everything at once

```
pytest test/python/whisper/test_whisper.py -v -s
```

---

## 5. Transcribe your own audio (standalone e2e)

`scripts/transcribe_whisper.py` drives the same greedy-decode harness the tests
use (shared from `test/python/whisper/whisper_infer.py`). It auto-prepares the model
(§3) on first run, so this one command works from a fresh checkout. Runs
identically in PowerShell and Git Bash.

If you've already run the tests (§4), the bundled **jfk.wav** sample is sitting
in the data dir — transcribe it directly:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav
```

(Expected: *"And so my fellow Americans, ask not what your country can do for
you..."*) The LibriSpeech clips fetched by §4 also work, e.g.
`test/python/data/whisper/librispeech/sample_0.wav`. For your own audio, pass any
16 kHz mono wav in place of the path above.

By default this prints the transcription **plus** the per-phase latency / RTF
table (see below), running one discarded warmup pass first so the numbers are
steady-state. For a bare transcription, turn those off:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav --no-metrics --no-warmup
```

Backend selection (default = GPU only): `--compare` runs **both** GPU and CPU
and prints a `CPU:` and a `GPU:` line for an apples-to-apples check; `--cpu` runs
**only** the CPU fp32 EP. The two are mutually exclusive.

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav --compare
```

- Audio must be **16 kHz mono**. Resample first if needed (e.g. `ffmpeg -i in.mp3
  -ar 16000 -ac 1 out.wav`). The feature extractor
  (`transformers.WhisperFeatureExtractor`) pads/truncates to Whisper's 30 s
  window — clips longer than 30 s need chunking (not handled here).
- `--max-length N` raises the decode-token cap (default 200; the static self-KV
  buffer caps it at 448) — bump it for clips that transcribe to long text.
- The GPU path requires the §2 environment (`THEROCK_DIST` + `PATH`); without it
  the script raises rather than silently falling back to CPU.

### Per-phase latency + RTF

The metrics table (on by default) reports encoder / prefill / decode latency,
total compute time, TTFT, decode tok/s, and **RTF** (real-time factor = compute
time ÷ audio duration; `< 1.0` is faster than real time). The default `--warmup`
runs one discarded GPU pass first so the numbers exclude the one-time model-DLL
load + kernel autotune; pass `--no-warmup` to see the cold-start cost instead.

Pipeline phases (all on GPU):

| Phase | Graph | What it does |
|---|---|---|
| encoder | `encoder_fixed.onnx` | mel `[1,128,3000]` → conv1d + 32 layers → `hidden_states [1,1500,1280]` + the cross-attention KV (computed once, constant for the whole generation) |
| prefill | `decoder_fixed_prefill.onnx` (S=4) | the 4 forced-start tokens in one shot → fills self-KV slots [0..3] → first generated token. This is the TTFT compute. |
| decode | `decoder_fixed_decode.onnx` (S=1) | one token per step: read self-KV + cross-KV, append one self-KV slot, argmax → next token, until `<|endoftext|>` or the 448-slot cap |


---

## 6. Compare backends (CPU / DirectML / Vulkan)

### MorphiZen vs CPU vs DirectML (same fp32 ONNX — the fair comparison)

```bash
pytest test/python/whisper/test_whisper.py::test_perf_decode_tps -v -s
```

Prints a decode-throughput table. Notes on what you'll see (gfx1151 reference):

| Backend | precision | decode tok/s | transcription |
|---|---|---|---|
| MorphiZen EP | fp32 ONNX | ~18 | JFK quote ✅ |
| CPU EP | fp32 ONNX | ~17 | JFK quote ✅ |
| DirectML EP | fp32 ONNX | **fails** | cross-attn `MultiHeadAttention` unsupported on DML |

**MorphiZen vs CPU is the only apples-to-apples pair** (identical fp32 ONNX, same
ORT API, same decode loop). DirectML **cannot run the Whisper decoder** — its
`com.microsoft.MultiHeadAttention` rejects the cross-attention layout — so there
is no DML number to compare.

### Vulkan (whisper.cpp) baseline

This is a **loose reference, not a like-for-like comparison**: whisper.cpp uses an
**f16 GGUF** model with its own hand-tuned C++ graph and KV management — a
different precision *and* a different computational graph from our fp32 ONNX.

One-time build (~10–20 min: fetches the Vulkan SDK + whisper.cpp + a ~3 GB f16
GGUF):

```bash
python scripts/build_whisper_vulkan.py
```

Run + transcribe the same clip:

```bash
install/whisper-vulkan/bin/whisper-cli.exe \
  -m install/whisper-vulkan/models/ggml-large-v3.bin \
  -f test/python/data/whisper/jfk.wav
```

Expected: the JFK quote, with a `ggml_vulkan: 0 = AMD Radeon ...` line in the log
confirming GPU dispatch. Its ~77 tok/s reflects f16 + a fully-fused native loop
with no Python/ORT per-step overhead — treat it as *"what a mature f16 Vulkan
stack achieves on the same GPU"*, not *"Vulkan is 4× faster than our EP."*

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `amdhip64_7.dll missing` / `Failed to link DLL` | `THEROCK_DIST` + `PATH` not set — see §2. |
| Transcription is garbage / a "GPU" run is suspiciously slow | Silent CPU fallback. Set the env (§2), clear the model cache (PowerShell `Remove-Item "$env:TEMP\morphizen_mlir_*"` / Git Bash `rm -f "$TEMP"/morphizen_mlir_*`), and re-run. Confirm GPU dispatch with `HIPDNN_EP_DEBUG=1` (look for `[REAL] wrap_*` lines on stderr). |
| Changed a runtime `.cpp` / kernel, behavior didn't change | Cached model DLLs embed the old bitcode. Clear the cache after rebuilding (PowerShell `Remove-Item "$env:TEMP\morphizen_mlir_*"` / Git Bash `rm -f "$TEMP"/morphizen_mlir_*`). |
| A test `skip`s with "audio unavailable" | The network can't reach github/HF for the test clips. Connect and re-run; the audio caches locally after the first fetch. |
| `pytest` can't find the EP / wrong ORT API | The EP DLL must match the pip `onnxruntime-directml` API version (see CLAUDE.md "ORT version must match pip"). Rebuild via `python build.py`. |

For internals (how the ONNX surgery works, the `no_causal` GQA path, the fp32
GQA enabling, known limitations), see the **Whisper gotchas** in
[CLAUDE.md](../CLAUDE.md).
