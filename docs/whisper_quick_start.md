<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper-large-v3 Quick Start

End-to-end guide for running **Whisper-large-v3** speech-to-text on the MorphiZen
EP from a fresh checkout: build the EP, build + compile the model, run the
tests, transcribe your own audio, and compare against CPU / Vulkan.

Whisper runs **fp16 by default** — it is both faster on GPU and bit-faithful: the
fp16 build keeps the `lm_head` in fp32 while the body is fp16, so greedy decoding
is argmax-lossless (verbatim transcription, prefill logit cosine 1.0 vs the
fp16 CPU reference). An **fp32** variant is also available (`--fp32`). **Both
precisions are built locally by one command** — `python scripts/build_whisper_models.py`
(a pinned OGA DirectML model builder in an isolated venv) — so there is no separate
opt-in build step. fp32 is selected at run time with `--fp32`; see §6 for
fp16-vs-fp32 numbers.

> **Shell:** commands in this guide are written for **Git Bash** (the repo's
> convention, same as the main [Quick Start](quick_start.md) — launch it from an
> "x64 Native Tools Command Prompt for VS"). The only commands that differ by
> shell are the environment exports in §2, which give a PowerShell equivalent.
> `pytest` / `git` / `python` commands work the same in either shell.

> **Paths in this guide** route the build tree and the install prefix through a
> single `$ROOT` you choose: the build tree at `$ROOT/build`, the install prefix
> at `$ROOT/local`. The EP DLL + `morphizen_config.json` land in `$ROOT/local/bin/`;
> the auto-downloaded TheRock SDK lives at `$ROOT/build/_therock/`. Set `$ROOT`
> once per shell (§1 / §2) and every later command reuses it.
>
> **Pick a SHORT `$ROOT` on Windows.** MSBuild's file tracker and the from-source
> LLVM build generate paths deep under `$ROOT/build/_deps/llvm-project-build/...`
> that bust the 260-char `MAX_PATH` limit if `$ROOT` is long — the failure is a
> cryptic `FileTracker : error FTK1011: could not create ... .tlog` mid-build, not
> a code error. A root like `C:\Users\you\work\hip-ep-workspace` (≈38 chars) is
> safe; building inside a deeply-nested worktree path is not. (Alternatively,
> enable Windows long-path support once as admin: `New-ItemProperty -Path
> "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled
> -Value 1 -PropertyType DWORD -Force`, then restart the shell.)

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

The test audio is fetched from public URLs at runtime (no `datasets` / `jiwer`
dependency), and the model build (`python scripts/build_whisper_models.py`, §3)
installs its pinned builder deps into a **dedicated isolated venv**
(`install/whisper-builder-venv/`), so it never touches the `hipdnn-ep` env or
shadows the OGA fork.

The one pip package you **do** need at a specific version is `onnxruntime` —
see §1b, which is mandatory before the EP will load.

---

## 1. Build the EP (one-time)

From the repo root. Choose a **short** `$ROOT` (see the MAX_PATH note above) and
route both the build tree and the install prefix through it.

**Git Bash:**

```bash
ROOT=/c/Users/$USER/work/hip-ep-workspace   # pick a SHORT path; see note above
python build.py --build_dir "$ROOT/build" --install_dir "$ROOT/local" --cmake_prefix_path "$ROOT/local"
```

**PowerShell:**

```powershell
$ROOT = "C:\Users\$env:USERNAME\work\hip-ep-workspace"   # pick a SHORT path; see note above
python build.py --build_dir "$ROOT\build" --install_dir "$ROOT\local" --cmake_prefix_path "$ROOT\local"
```

This resolves the LLVM/MLIR/Protobuf/FlatBuffers + ONNX Runtime deps (auto-downloaded
unless `--cmake_prefix_path` points at prebuilt ones) + the TheRock ROCm SDK,
detects your GPU, and builds the compiler + MorphiZen EP into `$ROOT/local/`. See the
main [Quick Start](quick_start.md) for details and troubleshooting.

After it finishes you should have `$ROOT/local/bin/onnxruntime_morphizen_ep.dll`.

---

## 1b. Install a matching ONNX Runtime wheel (mandatory)

**You must do this, or the EP will not load.** The EP DLL links the ORT version
pinned in `cmake/deps.txt` (currently **1.25.1**) and requests that exact ORT
C-API version when it registers. The Python tests `import onnxruntime`, so the
pip package must expose the **same** API version. A mismatch fails registration
(`The requested API version [N] is not available...`) and then segfaults
(Windows access violation on session create).

PyPI's `onnxruntime-directml` frequently **lags** the pinned tag (e.g. PyPI tops
out at 1.24.x while the repo pins 1.25.1), so `pip install onnxruntime-directml`
alone is usually **not** enough — you build a matching wheel from source (this is
exactly what CI does). Run from an **x64 Native Tools Command Prompt for VS**
(Ninja + MSVC required); `$ROOT` is the same short path as §1, and the
`v1.25.1` / PR-patch values come from `cmake/deps.txt` +
`.github/workflows/windows-build.yml` (`ONNXRUNTIME_VERSION` / `ONNXRUNTIME_PR_PATCHES`):

```bash
git clone --depth 1 --branch v1.25.1 --recurse-submodules --shallow-submodules \
  https://github.com/Microsoft/onnxruntime.git "$ROOT/source-onnxruntime"
cd "$ROOT/source-onnxruntime"
# Apply each PR listed in ONNXRUNTIME_PR_PATCHES (currently just 28608):
curl -L -o /tmp/ort.patch https://github.com/microsoft/onnxruntime/pull/28608.patch
git apply --whitespace=nowarn /tmp/ort.patch
# Build the shared lib + wheel (run build.bat from cmd / Native Tools prompt):
build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error \
  --skip_submodule_sync --build_dir "$ROOT\build-onnxruntime" --skip_tests \
  --disable_memleak_checker --use_dml --cmake_generator Ninja --build_wheel
pip install --force-reinstall "$ROOT/build-onnxruntime/Release/dist/onnxruntime_directml-1.25.1-*.whl"
python -c "import onnxruntime as ort; print(ort.__version__)"   # must print 1.25.1
```

The wheel is Python-version-specific (`cp314` for Python 3.14). **If PyPI has
caught up** to the pinned version, skip the source build and just
`pip install --force-reinstall onnxruntime-directml==1.25.1`.

---

## 2. Set up the runtime environment (every shell)

The EP **compiles and links the model into a GPU DLL at session init**, so the
ROCm runtime libraries must be on `PATH` and `THEROCK_DIST` must point at the SDK.
Run this in every shell before any Whisper command.

Use the same `$ROOT` you built with in §1. The auto-downloaded TheRock SDK lives
under the build dir (`$ROOT/build/_therock`); if you passed your own
`-DTHEROCK_DIST`, point at that instead. `MORPHIZEN_EP_BIN` tells the Python
tests where the EP DLL is — **required when you installed out-of-tree** (the
`$ROOT/local` layout here); the tests otherwise look in the legacy in-repo
`install/dist/bin` and `skip` with "MorphiZen EP not found".

**Git Bash** (the convention for the rest of this guide):

```bash
cd <repo-root>
conda activate hipdnn-ep
export ROOT=/c/Users/$USER/work/hip-ep-workspace   # same short path as §1
export THEROCK_DIST="$ROOT/build/_therock"
export MORPHIZEN_EP_BIN="$ROOT/local/bin"
export PATH="$THEROCK_DIST/bin:$ROOT/local/bin:$PATH"
```

**PowerShell** (equivalent — note `$env:` syntax and `;`-separated `Path`):

```powershell
cd <repo-root>
conda activate hipdnn-ep
$ROOT = "C:\Users\$env:USERNAME\work\hip-ep-workspace"   # same short path as §1
$env:THEROCK_DIST = "$ROOT\build\_therock"
$env:MORPHIZEN_EP_BIN = "$ROOT\local\bin"
$env:Path = "$env:THEROCK_DIST\bin;$ROOT\local\bin;$env:Path"
```

> **Why this matters:** without `THEROCK_DIST` / `PATH`, the EP fails to link the
> model DLL (`amdhip64_7.dll missing` / `Failed to link DLL`) and silently falls
> back to CPU. If a "GPU" run is suspiciously slow or wrong, check this first.
> Without `MORPHIZEN_EP_BIN` (out-of-tree install), the tests can't find the EP
> DLL and `skip` instead of running.

---

## 3. Build + compile the model

Acquisition is **two steps**. First, **build the raw models** — `python
scripts/build_whisper_models.py` builds BOTH the fp32 (`models/whisper-large-v3-onnx/`)
and fp16 (`models/whisper-large-v3-onnx-fp16/`) bundles from
`openai/whisper-large-v3` (pinned HF revision) via a pinned OGA DirectML model
builder running in an isolated venv. No manual stock-OGA install is needed — the
builder venv is self-contained. First run downloads HF weights + builds (~10 min);
idempotent after.

```
python scripts/build_whisper_models.py
```

Then **prepare the model for the EP** — `setup_whisper_model.py` is consume-only:
it applies the required ONNX surgery (`past_sequence_length` input + position-embed
/ token-embed fixes for the static shared-buffer KV cache) and runs `fix_shapes`
to lock the static shapes on the already-built raw bundle. It is idempotent and
instant; if the raw model is absent it prints the build hint above and exits.

```
python scripts/setup_whisper_model.py          # fp16 (default)
python scripts/setup_whisper_model.py --fp32   # fp32 model dir
```

Verify the variants were produced (`ls` on Git Bash, `dir` on PowerShell) — the
default fp16 bundle lives in `models/whisper-large-v3-onnx-fp16/`, the fp32 bundle
in `models/whisper-large-v3-onnx/`; both have the same file set:

```
models/whisper-large-v3-onnx-fp16/   (default; fp32 dir mirrors this layout)
  encoder.onnx (+.data), decoder.onnx (+.data), tokenizer.json, genai_config.json   (built raw bundle)
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

## 3b. Precisions (fp16 default, fp32 opt-out)

fp16 is the **default** precision — faster on GPU and bit-faithful (fp32 `lm_head`
keeps greedy argmax-lossless). Both bundles are produced **together** by the same
`python scripts/build_whisper_models.py` in §3 — the pinned OGA DirectML model
builder emits an fp16 body with an fp32 `lm_head` for the default model and an
all-fp32 model for `--fp32`. `python scripts/setup_whisper_model.py` (default) and
`--fp32` apply the SAME surgery + `fix_shapes`, writing
`models/whisper-large-v3-onnx-fp16/` and `models/whisper-large-v3-onnx/`.

No manual `onnxruntime-genai-directml` install is needed and the OGA fork is never
shadowed — the builder runs in its own isolated venv (see §0). The default
transcribe / test commands run fp16; add `--fp32` to use the fp32 model instead
(§5, §6).

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
pytest test/numeric/tests/test_whisper_encoder_attention.py test/numeric/tests/test_whisper_cross_attention.py test/numeric/tests/test_whisper_self_attention.py test/numeric/tests/test_conv1d.py test/numeric/tests/test_layer_norm.py --backend ort_ep --ep-name MorphiZenExecutionProvider --ep-dll $ROOT/local/bin/onnxruntime_morphizen_ep.dll --ep-option config_file=$ROOT/local/bin/morphizen_config.json -v
```

### 4e. MLIR conversion (LIT)

```
ctest --test-dir $ROOT/build -C Release -R MorphizenMLIRLitTests
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

The default run uses the fp16 model. Add `--fp32` to run the fp32 model instead
(build both precisions first with `python scripts/build_whisper_models.py`, §3).
The metrics label shows the precision:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav --fp32
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

### MorphiZen vs CPU vs DirectML, fp32 + fp16 (the fair comparison)

```bash
pytest test/python/whisper/test_whisper.py::test_perf_decode_tps -v -s
```

Prints a decode-throughput table across MorphiZen / CPU / DirectML for **both
precisions**. The fp32 rows always run; the fp16 rows run only if the fp16 model
is available (§3) and are skipped with a note otherwise. The DirectML leg is
skipped by default (it always fails on the Whisper decoder, below); opt in with
`HIPDNN_WHISPER_PERF_DML=1`. Reference numbers (gfx1151, jfk.wav ~11 s,
2026-06-10; numbers vary run-to-run):

| Backend | precision | decode tok/s | encoder ms | transcription |
|---|---|---|---|---|
| MorphiZen EP | **fp16** | **~45** | **~460** | JFK quote ✅ |
| MorphiZen EP | fp32 | ~24 | ~1300 | JFK quote ✅ |
| CPU EP | fp32 | ~22 | — | JFK quote ✅ |
| CPU EP | fp16 | ~18 | — | JFK quote ✅ (emulated fp16) |
| DirectML EP | fp32 / fp16 | **fails** | — | cross-attn `MultiHeadAttention` unsupported on DML |

The in-suite fp16 number (~45 tok/s) now matches the isolated
`scripts/transcribe_whisper.py` run (~45–48). *(Historically the perf test
under-reported fp16 at ~28: setting `HIPDNN_EP_DEBUG=1` to capture the dispatch
tripwire latched a process-wide debug flag — `static const bool`, see
`include/hip/debug_log.h` — that turned on per-op FD logging for the whole
process and throttled decode; pytest's `capfd` capture compounded it. The test
now never sets that env var and uses a bounded FD-redirect only around
session-create. **General rule: never enable tracing/debug env vars in a process
that will later measure throughput.**)*

**fp16 is both correct AND faster on MorphiZen here** — decode ~24→~45 tok/s and
encoder ~1300→~460 ms vs MorphiZen fp32, while still emitting the verbatim JFK
quote (argmax-lossless because the OGA build keeps `lm_head` fp32). MorphiZen-fp32
vs CPU-fp32 is the apples-to-apples pair (identical ONNX, same ORT API, same loop);
note run-to-run variance. CPU fp16 is *slower* than CPU fp32 (ORT CPU has no native
fp16 compute — it's emulated). DirectML **cannot run the Whisper decoder** in
either precision — `com.microsoft.MultiHeadAttention` rejects the cross-attention
layout — confirmed by a controlled A/B (same `decoder.onnx`, same inputs, same
call: runs on the **CPU EP**, aborts on the **DML EP** at the first cross-attn
node, only the provider differs). The DML *encoder* runs fine; only the decoder
cross-attn node is rejected. (Note: the model being OGA-`-e dml`-exported does
**not** imply DML-EP support — `-e dml` sets the activation dtype/layout, not op
coverage.) Re-run the perf test on your hardware for current figures.

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
stack achieves on the same GPU"* (~1.7× our fp16's ~45 tok/s), not a like-for-like
EP comparison.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `amdhip64_7.dll missing` / `Failed to link DLL` | `THEROCK_DIST` + `PATH` not set — see §2. |
| Transcription is garbage / a "GPU" run is suspiciously slow | Silent CPU fallback. Set the env (§2), clear the model cache (PowerShell `Remove-Item "$env:TEMP\morphizen_mlir_*"` / Git Bash `rm -f "$TEMP"/morphizen_mlir_*`), and re-run. Confirm GPU dispatch with `HIPDNN_EP_DEBUG=1` (look for `[REAL] wrap_*` lines on stderr). |
| Changed a runtime `.cpp` / kernel, behavior didn't change | Cached model DLLs embed the old bitcode. Clear the cache after rebuilding (PowerShell `Remove-Item "$env:TEMP\morphizen_mlir_*"` / Git Bash `rm -f "$TEMP"/morphizen_mlir_*`). |
| A test `skip`s with "audio unavailable" | The network can't reach github/HF for the test clips. Connect and re-run; the audio caches locally after the first fetch. |
| Every test `skip`s with "MorphiZen EP not found — run build.py first" | The tests can't locate the EP DLL. For an out-of-tree install (`$ROOT/local`), set `MORPHIZEN_EP_BIN` (§2). Verify the DLL exists at `$ROOT/local/bin/onnxruntime_morphizen_ep.dll`. |
| EP registration fails (`requested API version [N] is not available`) / access violation on session create | The pip `onnxruntime` version ≠ the ORT the EP links (`cmake/deps.txt`). PyPI's `onnxruntime-directml` often lags the pinned tag, so `pip install` alone won't fix it — build a matching ORT wheel from source and install it (see §1b). |

For internals (how the ONNX surgery works, the `no_causal` GQA path, the fp32
GQA enabling, known limitations), see the **Whisper gotchas** in
[CLAUDE.md](../CLAUDE.md).
