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
both default to fp16, and the per-variant smoke test runs fp16 only. (Pass
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

Per-variant EP correctness is covered by
`test/python/whisper/test_whisper_variant_smoke.py` — see [§4f](#4f-per-variant-smoke-turbo--small-sizes).

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

## 3. Get + compile the model

**You normally don't run anything here** — the tests (§4) and
`transcribe_whisper.py` (§5) prepare the model on demand: they download the raw
OGA bundle from the AMD HF repo on first use, then apply the EP surgery +
`fix_shapes` automatically. To do it ahead of time (or to verify the download),
run the consume-only setup wrapper, which fetches the raw bundle if absent and
prepares it for the EP:

```
python scripts/setup_whisper_model.py          # fp16 (default) — amd/whisper-large-v3-onnx-fp16
python scripts/setup_whisper_model.py --fp32   # fp32          — amd/whisper-large-v3-onnx-fp32
```

This downloads `encoder.onnx`/`decoder.onnx` (+ `.data`) + tokenizer + config from
Hugging Face (first run ~3–6 GB per precision), then applies the ONNX surgery
(`past_sequence_length` input + position-/token-embed fixes for the static
shared-buffer KV cache) and `fix_shapes`. Idempotent — instant once prepared.

> **Gated / rate-limited?** If the download fails for auth reasons, run
> `hf auth login` with a token that can read the repo and retry. If HF is
> unreachable entirely, build the models locally instead (§3b).

Verify the variants were produced (`ls` on Git Bash, `dir` on PowerShell) — the
default fp16 bundle lives in `models/whisper-large-v3-onnx-fp16/`, the fp32 bundle
in `models/whisper-large-v3-onnx/`; both have the same file set:

```
models/whisper-large-v3-onnx-fp16/   (default; fp32 dir mirrors this layout)
  encoder.onnx (+.data), decoder.onnx (+.data), tokenizer.json, genai_config.json   (raw bundle from HF)
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

## 3b. Build the models locally (backup)

The §3 download is the normal path. Build locally only if HF is unreachable, or
to **reproduce** the published models from source. `python
scripts/build_whisper_models.py` builds from `openai/whisper-large-v3` (pinned HF
revision) via a pinned OGA DirectML model builder running in an isolated venv into
`models/whisper-large-v3-onnx{,-fp16}/`. This is exactly how the AMD HF repos were
produced, so the output is byte-equivalent. First run downloads the HF *weights* +
builds (~10 min); idempotent after. The default precision is **fp16**; large-v3's
fp32-vs-fp32 cross-backend benchmark (§6) needs the fp32 bundle too, so pass
`--precision both` when reproducing large-v3 locally.

```
python scripts/build_whisper_models.py --variant large-v3 --precision both  # fp32 + fp16
python scripts/build_whisper_models.py --variant large-v3                    # fp16 only
```

Because this writes the same `models/whisper-large-v3-onnx{,-fp16}/` dirs the §3
download targets, a subsequent `setup_whisper_model.py` (or any test) sees the raw
bundle already present and **skips the download** — the local build pre-populates
the cache. No manual `onnxruntime-genai-directml` install is needed and the OGA
fork is never shadowed — the builder runs in its own isolated venv (see §0).

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

Clear the compiled-model cache to force a real compile, then run the test
(parametrized across fp16 + fp32 — split per precision on tight memory, see §4c):

```
# PowerShell:  Remove-Item "$env:TEMP\morphizen_mlir_*" -ErrorAction Ignore
# Git Bash:    rm -f "$TEMP"/morphizen_mlir_*
pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -k fp16 -v -s
pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -k fp32 -v -s
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

These three tests are parametrized across **both** precisions (`fp16` + `fp32`).
On memory-constrained machines, run each precision in its **own pytest process**
(`-k "... and fp16"`, then `-k "... and fp32"`) so the first precision's GPU
memory is fully reclaimed before the second starts:

```bash
SEL="encoder_correctness or decoder_prefill_correctness or decoder_decode_correctness"
pytest test/python/whisper/test_whisper.py -k "($SEL) and fp16" -v -s
pytest test/python/whisper/test_whisper.py -k "($SEL) and fp32" -v -s
```

> **Why split by precision?** The MorphiZen runtime keeps two **process-global**
> GPU caches that are NOT freed on `InferenceSession` destruction — the transient
> GPU buffer pool (`g_gpu_buffer_pool`) and the hipBLASLt autotune cache
> (`g_gemm_algo_cache`) — they live until the process exits. Running fp16 + fp32
> in one process therefore accumulates both precisions' GPU footprint (the fp32
> constants blob is ~2× the fp16 one). On a 32 GB UMA part this drives the peak
> high; splitting into one process per precision caps it at a single precision's
> footprint. To run both in one process anyway (plenty of memory), drop the
> `and fp16`/`and fp32` filters.

### 4d. Per-op numeric tests (vs ORT CPU)

Conv1d, attention, and LayerNorm, both fp16 and fp32 (one line):

```
pytest test/numeric/tests/test_whisper_encoder_attention.py test/numeric/tests/test_whisper_cross_attention.py test/numeric/tests/test_whisper_self_attention.py test/numeric/tests/test_conv1d.py test/numeric/tests/test_layer_norm.py --backend ort_ep --ep-name AMDGPUExecutionProvider --ep-dll $ROOT/local/bin/amdgpu-ep.dll --ep-option profile=llm --ep-option config_file=$ROOT/local/bin/morphizen_config.json -v
```

### 4e. MLIR conversion (LIT)

```
ctest --test-dir $ROOT/build -C Release -R MorphizenMLIRLitTests
```

### 4f. Per-variant smoke (turbo + small sizes)

§4a–4e all target **large-v3**. The other five variants (turbo + tiny/base/small/
medium) are covered by one parametrized smoke test that asserts **EP-GPU greedy
tokens == ORT-CPU greedy tokens** on the *same* variant. (It does NOT assert
verbatim text — the small models have low transcription accuracy, so EP-vs-CPU
token equality is what isolates EP correctness from model quality.)

First build + prepare the variants (see [Supported variants](#supported-variants)
for the build-all loop), then run the whole set:

```bash
pytest test/python/whisper/test_whisper_variant_smoke.py -v -s
```

…or one variant at a time:

```bash
pytest test/python/whisper/test_whisper_variant_smoke.py -k tiny -v -s
```

Each variant prints its GPU transcription and **passes** when it matches the CPU
reference token-for-token. A variant that has not been built/prepared, or a host
without the EP, **skips** cleanly. Verified on gfx1151 (bitcode path): all five
emit the verbatim JFK quote.

> **Run on the default bitcode path** — do NOT set `HIPEP_ARTIFACT_FORMAT=NATIVE`.
> The opt-in NATIVE (per-model-DLL) path predates the bitcode-JIT fix and can mis-
> decode (e.g. tiny emits EOT immediately); the default bitcode JIT is the correct,
> validated path.

### Run everything at once

```
pytest test/python/whisper/test_whisper.py -v -s                  # large-v3 full matrix
pytest test/python/whisper/test_whisper_variant_smoke.py -v -s    # turbo + small sizes (§4f)
```

On a 32 GB UMA machine, prefer running the precision-parametrized tests one
precision at a time (see §4c) instead of the all-at-once `test_whisper.py` command.
The variant smoke test is fp16-only and lightweight (the small models are tiny),
so it runs comfortably in one process.

---

## 5. Transcribe your own audio (standalone e2e)

`scripts/transcribe_whisper.py` drives the same greedy-decode harness the tests
use (shared from `test/python/whisper/whisper_infer.py`). It auto-prepares the model
(§3) on first run, so this one command works from a fresh checkout. Runs
identically in PowerShell and Git Bash.

Transcribe the bundled **jfk.wav** demo clip — if it isn't cached yet, the script
**downloads it on demand**, so this works on a fresh checkout with nothing
pre-fetched:

```
python scripts/transcribe_whisper.py test/python/data/whisper/jfk.wav
```

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
| Model setup fails / `Could not obtain the Whisper raw model` | The raw bundle download from `amd/whisper-large-v3-onnx-{fp16,fp32}` failed. If it's an auth / rate-limit error, run `hf auth login` and retry. If HF is unreachable, build the models locally instead (§3b: `python scripts/build_whisper_models.py`). |
| Every test `skip`s with "AMDGPU EP not found — run build.py first" | The tests can't locate the EP DLL. For an out-of-tree install (`$ROOT/local`), set `HIPEP_EP_BIN` (§2). Verify `amdgpu-ep.dll` (+ `hipep-backend.dll` + `hipep.dll`) exist at `$ROOT/local/bin/`. |
| EP registration fails (`requested API version [N] is not available`) / access violation on session create | The pip `onnxruntime` version ≠ the ORT the EP links (`cmake/deps.txt`). PyPI's `onnxruntime-directml` often lags the pinned tag, so `pip install` alone won't fix it — build a matching ORT wheel from source and install it (see §1b). |

For internals (how the ONNX surgery works, the `no_causal` GQA path, the fp32
GQA enabling, known limitations), see the **Whisper gotchas** in
[CLAUDE.md](../CLAUDE.md).
