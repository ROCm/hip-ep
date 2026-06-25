<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper-large-v3 Quick Start

End-to-end guide for running **Whisper** speech-to-text (large-v3 + the
turbo/tiny/base/small/medium variants) on the AMDGPU EP from a fresh checkout:
build the EP, build + prepare the model, run the tests, transcribe your own audio,
and compare against CPU / Vulkan.

Whisper runs **fp16 by default** — it is both faster on GPU and bit-faithful: the
fp16 build keeps the `lm_head` in fp32 while the body is fp16, so greedy decoding
is argmax-lossless (verbatim transcription, prefill logit cosine 1.0 vs the
fp16 CPU reference). An **fp32** variant is also available (`--fp32`). **Both
precisions are downloaded automatically from the AMD Hugging Face repos on first
use** — [`amd/whisper-large-v3-onnx-fp16`](https://huggingface.co/amd/whisper-large-v3-onnx-fp16)
and [`amd/whisper-large-v3-onnx-fp32`](https://huggingface.co/amd/whisper-large-v3-onnx-fp32) —
so the tests and `transcribe_whisper.py` work from a fresh checkout with no
separate model step. Building the models locally is the **backup** (§3b). fp32 is
selected at run time with `--fp32`; see §6 for fp16-vs-fp32 numbers.

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

## Supported variants

The EP supports these Whisper variants (all share one architecture + decoder
surgery; only shape params, n_mels, vocab, and special-token IDs differ):

| Variant        | n_mels | enc / dec layers | heads | vocab | source                        |
|----------------|--------|------------------|-------|-------|-------------------------------|
| large-v3       | 128    | 32 / 32          | 20    | 51866 | AMD HF (auto-download)        |
| large-v3-turbo | 128    | 32 / 4           | 20    | 51866 | local OGA build               |
| medium         | 80     | 24 / 24          | 16    | 51865 | local OGA build               |
| small          | 80     | 12 / 12          | 12    | 51865 | local OGA build               |
| base           | 80     | 6 / 6            | 8     | 51865 | local OGA build               |
| tiny           | 80     | 4 / 4            | 6     | 51865 | local OGA build               |

`head_dim` is 64 and the decoder context is 448 for every variant.

Build + prepare a single non-large-v3 variant (large-v3 auto-downloads on first
use; the others require a local OGA build):

```bash
python scripts/build_whisper_models.py --variant large-v3-turbo   # or tiny/base/small/medium (fp16)
python scripts/setup_whisper_model.py --variant large-v3-turbo    # surgery + fix_shapes (fp16)
```

Or **build + prepare ALL five local variants at once** (large-v3 is fetched
on-demand by the test, so it is not in this list):

```bash
python scripts/build_whisper_models.py                            # default set, fp16
for v in large-v3-turbo tiny base small medium; do
  python scripts/setup_whisper_model.py --variant "$v"            # surgery + fix_shapes
done
```

The added variants only need **fp16** — the build script and `setup_whisper_model.py`
both default to fp16, and the per-variant test legs run fp16 only. (Pass
`--precision both` / `--fp32` if you ever want fp32 for one of these.) This halves
build + disk + test work; large-v3 keeps its fp32 bundle for the cross-backend
fp32-vs-fp32 benchmark (§6).

`--variant` accepts a comma-separated list or multiple `--variant` flags; omitting
it builds the default set (`large-v3-turbo tiny base small medium`). See
`python scripts/build_whisper_models.py --list` for the full list.

> **large-v3-turbo build note.** Turbo has an asymmetric encoder/decoder (32 / 4
> layers); the stock OGA 0.13.1 builder crashes on it. `build_whisper_models.py`
> monkeypatches the builder in its isolated venv to fix this — no action needed,
> but if you build turbo by some other means, expect an `IndexError: index 4 is
> out of range`.

Per-variant EP correctness + performance is covered by
`test/python/whisper/test_whisper.py` (fp16 for every variant + fp32 for
large-v3) — see [§4](#4-run-the-tests).

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
dependency), and the Whisper models are **downloaded from Hugging Face** on first
use (§3) via `huggingface_hub` (already in `environment.yml`). The optional local
model build (`python scripts/build_whisper_models.py`, §3b) installs its pinned
builder deps into a **dedicated isolated venv** (`install/whisper-builder-venv/`),
so it never touches the `hipdnn-ep` env or shadows the OGA fork.

The one pip package you **do** need at a specific version is `onnxruntime` —
see §1b, which is mandatory before the EP will load.

---

## 1. Build the EP (one-time)

From the repo root. Choose a **short** `$ROOT` (see the MAX_PATH note above) and
route the build trees + install prefixes through it. Run the LLVM step from an
**x64 Native Tools Command Prompt for VS 2022** (Ninja needs the MSVC env).

### Step 1a — build LLVM into an `llvm-install` prefix (Tier-1, one-time)

**Build the EP against a real `llvm-install` (Tier-1), not the from-source
fallback.** On Windows a Tier-2 / FetchContent in-tree LLVM produces a `hipep.dll`
whose in-process **bitcode** JIT crashes at session-create; CI builds and
`find_package()`s a standalone `llvm-install`, and matching that locally is what
makes the default bitcode path work. The recipe (pins live in
`.github/workflows/windows-build.yml`):

```bash
ROOT=/c/Users/$USER/work/hip-ep-workspace   # pick a SHORT path; see note above
git clone --depth 1 -b llvmorg-22.1.0 https://github.com/llvm/llvm-project.git "$ROOT/llvm-src"
cmake -G Ninja -S "$ROOT/llvm-src/llvm" -B "$ROOT/build-llvm" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
  -DCMAKE_INSTALL_PREFIX="$ROOT/llvm-install" -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_TOOLS=ON -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake --build "$ROOT/build-llvm" --target install
```

Multi-hour the first time; reusable forever after — point every `build.py` at it.

### Step 1b — build the EP against `llvm-install`

```bash
python build.py --build_dir "$ROOT/build" --install_dir "$ROOT/local" --cmake_prefix_path "$ROOT/llvm-install"
```

PowerShell: same commands with `$ROOT = "C:\Users\$env:USERNAME\work\hip-ep-workspace"`
and `\`-separated paths.

`build.py` resolves the remaining deps (Protobuf/FlatBuffers + ONNX Runtime
auto-downloaded), the TheRock ROCm SDK, detects your GPU, and builds the compiler
+ the EP backend (`hipep.dll`) into `$ROOT/local/`. See the main
[Quick Start](quick_start.md) for details and troubleshooting.

After it finishes you should have `$ROOT/local/bin/hipep.dll`.

> **AMDGPU umbrella EP.** The tests reach the backend through the AMD GPU
> umbrella EP — `amdgpu-ep.dll` (registration name `AMDGPUExecutionProvider`),
> which loads `hipep-backend.dll` → `hipep.dll`. The umbrella + shim are built
> from the `onnxruntime-ep-amdgpu` fork (see `.github/workflows/windows-build.yml`
> for the pinned commit + `cmake -DUSE_AMDGPU=ON` recipe) and must sit next to
> `hipep.dll` in `$ROOT/local/bin/`. The umbrella selects the hipep backend via
> the `profile=llm` provider option.

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
(Ninja + MSVC required; `build.bat` is a Windows batch script). `$ROOT` is the
same short path as §1, and the `v1.25.1` / PR-patch values come from
`cmake/deps.txt` + `.github/workflows/windows-build.yml`
(`ONNXRUNTIME_VERSION` / `ONNXRUNTIME_PR_PATCHES`).

**PowerShell:**

```powershell
git clone --depth 1 --branch v1.25.1 --recurse-submodules --shallow-submodules `
  https://github.com/Microsoft/onnxruntime.git "$ROOT\source-onnxruntime"
cd "$ROOT\source-onnxruntime"
# Apply each PR listed in ONNXRUNTIME_PR_PATCHES (currently just 28608):
Invoke-WebRequest https://github.com/microsoft/onnxruntime/pull/28608.patch -OutFile "$env:TEMP\ort.patch"
git apply --whitespace=nowarn "$env:TEMP\ort.patch"
# Build the shared lib + wheel:
.\build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error `
  --skip_submodule_sync --build_dir "$ROOT\build-onnxruntime" --skip_tests `
  --disable_memleak_checker --use_dml --cmake_generator Ninja --build_wheel
$wheel = (Get-ChildItem "$ROOT\build-onnxruntime\Release\dist\onnxruntime_directml-1.25.1-*.whl" | Select-Object -First 1).FullName
pip install --force-reinstall "$wheel"
python -c "import onnxruntime as ort; print(ort.__version__)"   # must print 1.25.1
```

**Git Bash** (run `build.bat` from a Native Tools prompt, or `cmd //c build.bat ...`):

```bash
git clone --depth 1 --branch v1.25.1 --recurse-submodules --shallow-submodules \
  https://github.com/Microsoft/onnxruntime.git "$ROOT/source-onnxruntime"
cd "$ROOT/source-onnxruntime"
curl -L -o /tmp/ort.patch https://github.com/microsoft/onnxruntime/pull/28608.patch
git apply --whitespace=nowarn /tmp/ort.patch
cmd //c "build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir \"$ROOT\\build-onnxruntime\" --skip_tests --disable_memleak_checker --use_dml --cmake_generator Ninja --build_wheel"
pip install --force-reinstall "$ROOT"/build-onnxruntime/Release/dist/onnxruntime_directml-1.25.1-*.whl
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
`-DTHEROCK_DIST`, point at that instead. `HIPEP_EP_BIN` tells the Python
tests where the EP DLL is — **required when you installed out-of-tree** (the
`$ROOT/local` layout here); the tests otherwise look in the legacy in-repo
`install/dist/bin` and `skip` with "AMDGPU EP not found".

**Git Bash** (the convention for the rest of this guide):

```bash
cd <repo-root>
conda activate hipdnn-ep
export ROOT=/c/Users/$USER/work/hip-ep-workspace   # same short path as §1
export THEROCK_DIST="$ROOT/build/_therock"
export HIPEP_EP_BIN="$ROOT/local/bin"
export PATH="$THEROCK_DIST/bin:$ROOT/local/bin:$PATH"
```

**PowerShell** (equivalent — note `$env:` syntax and `;`-separated `Path`):

```powershell
cd <repo-root>
conda activate hipdnn-ep
$ROOT = "C:\Users\$env:USERNAME\work\hip-ep-workspace"   # same short path as §1
$env:THEROCK_DIST = "$ROOT\build\_therock"
$env:HIPEP_EP_BIN = "$ROOT\local\bin"
$env:Path = "$env:THEROCK_DIST\bin;$ROOT\local\bin;$env:Path"
```

> **Why this matters:** without `THEROCK_DIST` / `PATH`, the EP fails to link the
> model DLL (`amdhip64_7.dll missing` / `Failed to link DLL`) and silently falls
> back to CPU. If a "GPU" run is suspiciously slow or wrong, check this first.
> Without `HIPEP_EP_BIN` (out-of-tree install), the tests can't find the EP
> DLL and `skip` instead of running.

> **Optional — `HIPEP_ARTIFACT_FORMAT=NATIVE` (escape hatch):** the tests default
> to the production bitcode (in-process LLVM-IR JIT) path, which works once you
> build against the Tier-1 `llvm-install` (Step 1a). `HIPEP_ARTIFACT_FORMAT=NATIVE`
> forces a per-model DLL instead; leave it unset for normal runs and in CI.

---

## 3. Get + prepare the model

**Sourcing differs by variant** — only large-v3 is published as a prebuilt ONNX:

| Variant | Source | Precisions |
|---|---|---|
| large-v3 | auto-download (AMD HF) | fp16 + fp32 |
| large-v3-turbo, tiny, base, small, medium | local OGA build (§3b) | fp16 |

**You normally don't run anything here** — the tests (§4) and
`transcribe_whisper.py` (§5) prepare the model on demand. To prepare ahead of time,
run the consume-only setup wrapper for a variant. It applies the EP surgery
(`past_sequence_length` input + position-/token-embed fixes for the static
shared-buffer KV cache) + `fix_shapes`; for large-v3 it first downloads the raw
bundle from HF (~3–6 GB/precision), for the other variants it consumes the bundle
you built in §3b. Idempotent.

```
python scripts/setup_whisper_model.py --variant large-v3          # fp16 (default), auto-downloads
python scripts/setup_whisper_model.py --variant large-v3 --fp32   # fp32, auto-downloads
python scripts/setup_whisper_model.py --variant tiny              # fp16; build it first (§3b)
```

`--variant` defaults to `large-v3`. Prepare all six at once (after building the
five local ones, §3b):

```bash
for v in large-v3 large-v3-turbo tiny base small medium; do
  python scripts/setup_whisper_model.py --variant "$v"
done
```

> **Gated / rate-limited large-v3 download?** Run `hf auth login` with a token that
> can read the repo and retry. If HF is unreachable, build large-v3 locally (§3b).

Each prepared variant dir (`models/whisper-<variant>-onnx[-fp16]/`) holds:

```
  encoder.onnx (+.data), decoder.onnx (+.data), tokenizer.json, genai_config.json   (raw bundle)
  encoder_fixed.onnx, decoder_surgery.onnx,
  decoder_fixed_prefill.onnx (S=4), decoder_fixed_decode.onnx (S=1)                  (surgered + fixed-shape)
```

> **First run per model is slower:** the EP JIT-compiles each `*_fixed*.onnx`
> in-process at session load (ONNX → HIP → LLVM IR); there is no on-disk model
> artifact to manage. After rebuilding the EP (runtime/kernel change) the next run
> JITs the new code automatically. (The opt-in `HIPEP_ARTIFACT_FORMAT=NATIVE` path
> instead writes a per-model DLL under `%TEMP%/morphizen_mlir_*`, which you clear
> with `rm -f "$TEMP"/morphizen_mlir_*` after a runtime change; the default JIT
> path needs no such step.)

---

## 3b. Build the variants locally

The five non-large-v3 variants (turbo + tiny/base/small/medium) are **not
published** — build them with the pinned OGA DirectML model builder (runs in an
isolated venv from `openai/whisper-<size>` at a pinned HF revision; the builder
also patches OGA's asymmetric-layer bug so turbo's 32-enc/4-dec model builds).
Default precision is **fp16** (all the per-variant tests run fp16):

```
python scripts/build_whisper_models.py                          # default set: turbo + tiny/base/small/medium (fp16)
python scripts/build_whisper_models.py --variant tiny,base      # specific variants
python scripts/build_whisper_models.py --list                   # show variant -> output dir
```

Then prepare each (surgery + `fix_shapes`, §3):

```bash
for v in large-v3-turbo tiny base small medium; do
  python scripts/setup_whisper_model.py --variant "$v"
done
```

large-v3 normally downloads (§3); to **reproduce it locally** instead (e.g. HF
unreachable, or to rebuild the fp32 bundle the §6 benchmark needs), pass
`--variant large-v3 --precision both`. The build writes the same
`models/whisper-large-v3-onnx{,-fp16}/` dirs the download targets, so a later
`setup_whisper_model.py` / test sees the bundle present and skips the download.
First build per variant downloads the HF *weights* + builds (~a few min each);
idempotent after. No manual `onnxruntime-genai-directml` install is needed and the
OGA fork is never shadowed — the builder runs in its own isolated venv (see §0).

### Precisions (fp16 default, fp32 opt-out)

fp16 is the **default** precision — faster on GPU and bit-faithful (fp32 `lm_head`
keeps greedy argmax-lossless). The default transcribe / test commands run fp16;
add `--fp32` to use the fp32 model instead (§5, §6). Both the HF download (§3) and
the local build (above) provide the same two bundles; `setup_whisper_model.py`
(default) and `--fp32` apply the SAME surgery + `fix_shapes` to each.

---

## 4. Run the tests

The test audio (jfk.wav from whisper.cpp, LibriSpeech clips from the HF
datasets-server) auto-downloads on first run; tests `skip` cleanly if the network
is unreachable.

### 4a. End-to-end transcription (the headline)

Runs the full pipeline on GPU and asserts the transcription matches the ORT CPU
fp32 reference **verbatim**:

The e2e test is parametrized over (variant, precision): fp16 for every variant +
fp32 for large-v3. Run the whole set, or one combo via the test-id substring:

```
pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -v -s
# one combo:
pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -k large-v3-fp16 -v -s
```

Expected: each (variant, precision) GPU run matches its CPU greedy tokens; large-v3
prints *"And so my fellow Americans, ask not what your country can do for you, ask
what you can do for your country."* and the test **passes**. (First run per model
is slow — it pays the MLIR compile of the encoder + both decoder variants.)

### 4b. Multi-clip correctness + accuracy (WER)

5 LibriSpeech clips + a long concatenated clip. Each asserts **GPU == CPU
verbatim** *and* **WER vs ground truth** within threshold:

```bash
pytest test/python/whisper/test_whisper.py -k "librispeech or long_30s" -v -s
```

### 4c. Per-phase correctness (all variants)

The three phase tests (encoder / prefill / decode cosine) and the e2e test are
parametrized over **(variant, precision)**: **fp16 across every variant** (turbo +
tiny/base/small/medium + large-v3) **plus a single fp32 leg on large-v3**. Run the
whole matrix in one process:

```bash
SEL="encoder_correctness or decoder_prefill_correctness or decoder_decode_correctness"
pytest test/python/whisper/test_whisper.py -k "$SEL" -v -s
```

…or narrow to one variant / precision with the test-id substrings (e.g.
`-k "$SEL and tiny-fp16"`, `-k "$SEL and large-v3-fp32"`).

> **Run the whole matrix in one process.** The dev/CI host is 128 GB, so fp16 (all
> variants) + fp32 (large-v3) fit comfortably together — no per-precision split. If
> the in-process bitcode JIT ever flakes on your box (a `0xc0000005` at
> session-create), build against the Tier-1 `llvm-install` (§1), or fall back to
> `HIPEP_ARTIFACT_FORMAT=NATIVE` (§4a escape hatch) — that bypasses the JIT
> entirely. Do not split by precision.

### 4d. Per-op numeric tests (vs ORT CPU)

Conv1d, attention, and LayerNorm — **fp16 + fp32 across every variant's shapes**
(attention parametrizes d_model/num_heads per variant; conv1d covers each
variant's encoder front-end). One line — note `profile=llm` is the ONLY provider
option the AMDGPU umbrella accepts (do NOT pass `config_file=`, the umbrella
rejects unknown provider options). `THEROCK_DIST` must be set so the numeric
backend can find the ROCm runtime DLLs:

```
pytest test/numeric/tests/test_whisper_encoder_attention.py test/numeric/tests/test_whisper_cross_attention.py test/numeric/tests/test_whisper_self_attention.py test/numeric/tests/test_conv1d.py test/numeric/tests/test_layer_norm.py --backend ort_ep --ep-name AMDGPUExecutionProvider --ep-dll $ROOT/local/bin/amdgpu-ep.dll --ep-option profile=llm -v
```

### 4e. MLIR conversion (LIT)

```
ctest --test-dir $ROOT/build -C Release -R MorphizenMLIRLitTests
```

### 4f. Per-variant coverage + the PERF line

All whisper tests live in `test_whisper.py`. §4a (e2e) and §4c (per-phase cosine)
are parametrized over **every variant** (fp16) **+ large-v3 (fp32)**, so a
variant's encoder/prefill/decode correctness and GPU==CPU greedy-token check are
covered there. Build the variants first (see
[Supported variants](#supported-variants)), then narrow to one with the test-id
substring, e.g.:

```bash
pytest "test/python/whisper/test_whisper.py::test_e2e_transcription_greedy" -k tiny -v -s
```

`test_e2e_transcription_greedy` also prints a steady-state **PERF** line per
(variant, precision) — encoder ms / prefill ms / decode tok/s / RTF, measured on
cached (already-compiled) sessions — so variant performance is recorded in CI logs
alongside the correctness check. For an isolated measurement use
`scripts/transcribe_whisper.py --variant <name>` (§5).

> **Both artifact formats produce identical tokens.** Default is the in-process
> bitcode JIT; `HIPEP_ARTIFACT_FORMAT=NATIVE` (per-model DLL) is the escape hatch
> if the JIT flakes on your host. Both are correct when the EP is built against a
> Tier-1 `llvm-install` (§1) and rebuilt after any runtime/kernel change (the
> NATIVE per-model DLL embeds the runtime bitcode at build time, so always rebuild
> + clear the cache after touching runtime code).

### Run everything at once

```
pytest test/python/whisper/test_whisper.py -v -s    # all variants fp16 + large-v3 fp32 + perf/WER/librispeech
```

The full `test_whisper.py` matrix (all variants fp16 + large-v3 fp32) runs in a
single process on the 128 GB dev/CI host — no per-precision split needed (see §4c).
The `-s` flag surfaces the per-variant `PERF` lines.

---

## 5. Transcribe your own audio (standalone e2e)

`scripts/transcribe_whisper.py` drives the same greedy-decode harness the tests
use (shared from `test/python/whisper/whisper_infer.py`). It runs **one** variant
per invocation — `--variant` (default `large-v3`); it auto-prepares large-v3 (§3)
on first run, so the default command works from a fresh checkout, while the other
variants must be built first (§3b). Runs identically in PowerShell and Git Bash.

Transcribe the bundled **jfk.wav** demo clip with the default large-v3 — if the
clip isn't cached yet, the script **downloads it on demand**, so this works on a
fresh checkout with nothing pre-fetched:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav                       # large-v3 (default)
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav --variant tiny        # a specific variant
```

It does **not** loop over all variants — pick one with `--variant` (to sweep all,
call it once per variant in a shell loop).

(Expected: *"And so my fellow Americans, ask not what your country can do for
you..."*) The LibriSpeech clips fetched by §4 also work, e.g.
`test/python/data/whisper/librispeech/sample_0.wav`. For your own audio, pass any
16 kHz mono wav in place of the path above (only the bundled jfk.wav is
auto-downloaded; other paths must exist).

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
(it auto-downloads on first use, same as fp16; see §3). The metrics label shows
the precision:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav --fp32
```

With `--metrics` (default on), `--variant <name>` is the standalone way to measure
a variant's decode tok/s / RTF — the same numbers the §4f `PERF` line prints in CI.

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

This whole section compares **Whisper-large-v3 only** — the cross-backend perf
test runs the one canonical model so every backend executes the *same* ONNX graph
(the variant perf numbers live in the per-variant `PERF` line, §4f).

### AMDGPU EP vs CPU vs DirectML — Whisper-large-v3, fp32 + fp16

```bash
pytest test/python/whisper/test_whisper.py::test_perf_decode_tps -v -s
```

Prints a decode-throughput table for **Whisper-large-v3** across the AMDGPU EP /
CPU / DirectML for **both precisions**. The fp32 rows always run; the fp16 rows run
only if the fp16 model is available (§3) and are skipped with a note otherwise. The
DirectML leg is skipped by default (it always fails on the Whisper decoder, below);
opt in with `HIPDNN_WHISPER_PERF_DML=1`. Reference numbers (gfx1151, jfk.wav ~11 s;
numbers vary run-to-run):

| Backend | precision | decode tok/s | encoder ms | transcription |
|---|---|---|---|---|
| AMDGPU EP | **fp16** | **~45** | **~460** | JFK quote ✅ |
| AMDGPU EP | fp32 | ~24 | ~1300 | JFK quote ✅ |
| CPU EP | fp32 | ~22 | — | JFK quote ✅ |
| CPU EP | fp16 | ~18 | — | JFK quote ✅ (emulated fp16) |
| DirectML EP | fp32 / fp16 | **fails** | — | cross-attn `MultiHeadAttention` unsupported on DML |

The in-suite fp16 number (~45 tok/s) matches the isolated
`scripts/transcribe_whisper.py` run (~45–48). The perf test never sets
`HIPDNN_EP_DEBUG` and captures the dispatch tripwire with a bounded FD-redirect
only around session-create, so decode throughput isn't throttled. **General rule:
never enable tracing/debug env vars in a process that will later measure
throughput** (some are latched process-wide — `static const bool`, see
`include/hip/debug_log.h`).

**fp16 is both correct AND faster on the AMDGPU EP here** — decode ~24→~45 tok/s
and encoder ~1300→~460 ms vs fp32, while still emitting the verbatim JFK quote
(argmax-lossless because the OGA build keeps `lm_head` fp32). AMDGPU-EP-fp32 vs
CPU-fp32 is the apples-to-apples pair (identical ONNX, same ORT API, same loop);
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
| Model setup fails / `Could not obtain the Whisper raw model` | The raw bundle download from `amd/whisper-large-v3-onnx-{fp16,fp32}` failed. If it's an auth / rate-limit error, run `hf auth login` and retry. If HF is unreachable, build the models locally instead (§3b: `python scripts/build_whisper_models.py`). |
| Every test `skip`s with "AMDGPU EP not found — run build.py first" | The tests can't locate the EP DLL. For an out-of-tree install (`$ROOT/local`), set `HIPEP_EP_BIN` (§2). Verify `amdgpu-ep.dll` (+ `hipep-backend.dll` + `hipep.dll`) exist at `$ROOT/local/bin/`. |
| EP registration fails (`requested API version [N] is not available`) / access violation on session create | The pip `onnxruntime` version ≠ the ORT the EP links (`cmake/deps.txt`). PyPI's `onnxruntime-directml` often lags the pinned tag, so `pip install` alone won't fix it — build a matching ORT wheel from source and install it (see §1b). |

For internals (how the ONNX surgery works, the `no_causal` GQA path, the fp32
GQA enabling, known limitations), see the **Whisper gotchas** in
[CLAUDE.md](../CLAUDE.md).
