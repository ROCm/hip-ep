<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# DFlash on HIP EP

## Scope and assumptions

This integration assumes:

1. The DFlash draft model runs on the CPU.
2. The target model runs through OGA with HIP EP on the AMD GPU.
3. The speculative-decoding controller and acceptance loop run in Python/PyTorch.

The target supports two accepted-prefix recovery methods:

- **Windowed-state crop:** the efficient path for supported hybrid models. After
partial acceptance, the controller crops the KV/position state and promotes
the accepted recurrent-state window slot without replaying the accepted
prefix.
- `replay_append`**:** the required compatibility path for
non-hybrid models. It rewinds the target and appends the accepted prefix
again. Windowed models can also use this path with `--force-replay` for
correctness and performance comparisons.

## 0. Prerequisites

### Default: reproduce the packaged runtime bundle

Use this path for the DFlash setup. It reproduces the same combined
runtime layout used by the validated runs. It builds into a new
directory and does not overwrite an existing HIP EP install, Conda environment,
or model directory.

Set the paths once:

```bash
export RUNTIME_ROOT="$PWD/dflash_runtime_gfx1151"
export ROCM_ROOT=/opt/rocm/core-7.11
export CONDA_ENV=dflash_hipep_state_window_repro
export CONDA="$(type -P conda)"
test -n "$CONDA" || {
  echo "Conda executable was not found on PATH; activate/load Miniforge first." >&2
  exit 1
}
export HIP_EP_REPO_ROOT="$PWD"
mkdir -p "$RUNTIME_ROOT"
```

### 1. Build the shared ONNX Runtime dependency

Build ONNX Runtime 1.31 first. Both HIP EP and OGA will use this exact ORT installation; keeping one ORT version prevents library mismatches:

```bash
git clone --branch v1.31.0 --depth 1 --recurse-submodules \
  https://github.com/microsoft/onnxruntime.git \
  "$RUNTIME_ROOT/source-onnxruntime"
```

Create the fresh Conda environment used for every build and test command:

```bash
"$CONDA" create -n "$CONDA_ENV" python=3.14 pip -y
"$CONDA" run -n "$CONDA_ENV" python -m pip install --upgrade pip
"$CONDA" run -n "$CONDA_ENV" python -m pip install \
  ninja numpy packaging setuptools wheel cmake lit filelock pyyaml
```

Continue building ORT inside that environment:

```bash
cd "$RUNTIME_ROOT/source-onnxruntime"
"$CONDA" run -n "$CONDA_ENV" ./build.sh \
  --config Release \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --build_dir "$RUNTIME_ROOT/build-onnxruntime" \
  --skip_tests \
  --cmake_generator Ninja \
  --build_wheel \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

cmake --install "$RUNTIME_ROOT/build-onnxruntime/Release" \
  --prefix "$RUNTIME_ROOT/ort-install"
mkdir -p "$RUNTIME_ROOT/ort-install/wheels"
ORT_WHEEL=$(find "$RUNTIME_ROOT/build-onnxruntime" \
  -name 'onnxruntime-*.whl' -type f -print -quit)
test -n "$ORT_WHEEL"
cp "$ORT_WHEEL" "$RUNTIME_ROOT/ort-install/wheels/"
```

Now build HIP EP from the windowed-state revision. The
`--cmake_prefix_path` option below tells HIP EP to link against the exact ORT
installation created in the previous step:

```bash
git clone https://github.com/ROCm/hip-ep.git "$RUNTIME_ROOT/hip-ep"
cd "$RUNTIME_ROOT/hip-ep"
git fetch https://github.com/devledas/hip-ep.git \
  devledas/state-window-hybrid-ops
git checkout FETCH_HEAD

"$CONDA" run -n "$CONDA_ENV" --no-capture-output python3 build.py \
  --config Release \
  --build_dir "$RUNTIME_ROOT/build-hip-ep" \
  --install_dir "$RUNTIME_ROOT/hip-install" \
  --cmake_generator Ninja \
  --cmake_prefix_path "$RUNTIME_ROOT/ort-install" \
  --therock_dist "$ROCM_ROOT" \
  --hip_arch gfx1151 \
  --skip_submodule_sync \
  --skip_tests \
  -j 8
```

Build OGA 0.17 against the same ORT install. Apply the OGA patch set required
by the matching OGA source revision; do not mix OGA builds against a different
ORT version:

```bash
git clone --branch v0.17.0 --recurse-submodules \
  https://github.com/microsoft/onnxruntime-genai.git \
  "$RUNTIME_ROOT/onnxruntime-genai"
cd "$RUNTIME_ROOT/onnxruntime-genai"
```

The four diffs below are upstream OGA pull requests. The v0.17.0 tag does not
contain all of the AMDGPU continuous-decoding, multimodal state forwarding, and
windowed-state APIs required by this integration. The diffs add those OGA-side
changes; they do not modify HIP EP or the DFlash controller. They are downloaded
into `/tmp` only as temporary patch files:

```bash
cd "$RUNTIME_ROOT/onnxruntime-genai"
test -z "$(git status --porcelain)" || {
  echo "OGA source is not clean; move it aside and create a fresh v0.17.0 checkout before applying these diffs." >&2
  exit 1
}

for pr in 2685 2691 2705 2712; do
  curl -fsSL "https://github.com/microsoft/onnxruntime-genai/pull/${pr}.diff" \
    -o "/tmp/oga-pr-${pr}.diff"
  git apply --whitespace=nowarn --3way \
    --exclude='test/*' --exclude='.github/*' \
    "/tmp/oga-pr-${pr}.diff"
done
```

```bash
cd "$RUNTIME_ROOT/onnxruntime-genai"

# Add the HIP EP repository's Python binding for windowed-state crop.
git apply --whitespace=nowarn \
  "$HIP_EP_REPO_ROOT/tools/dflash-hipep/patches/oga-v0.17-window-crop-python-multimodal.patch"

# Connect OGA's AMDGPU device interface to the HIP EP batched state-slot-copy
# API so each partial commit uses one ordered GPU kernel.
git apply --whitespace=nowarn \
  "$HIP_EP_REPO_ROOT/tools/dflash-hipep/patches/oga-v0.17-amdgpu-batched-state-slot-copy.patch"

# Forward OGA snapshot/rewind through the multimodal decoder so recurrent
# state is included in the legacy rewind/replay path.
git apply --whitespace=nowarn \
  "$HIP_EP_REPO_ROOT/tools/dflash-hipep/patches/oga-main-5c19002-amdgpu-multimodal-snapshot.patch"

rm -rf "$RUNTIME_ROOT/oga-ort-home"
mkdir -p "$RUNTIME_ROOT/oga-ort-home/include" "$RUNTIME_ROOT/oga-ort-home/lib"
cp -r "$RUNTIME_ROOT/ort-install/include/onnxruntime/." \
  "$RUNTIME_ROOT/oga-ort-home/include/"
cp -a "$RUNTIME_ROOT"/ort-install/lib/libonnxruntime.so* \
  "$RUNTIME_ROOT/oga-ort-home/lib/"

"$CONDA" run -n "$CONDA_ENV" python -m pip install -r requirements-dev.txt
"$CONDA" run -n "$CONDA_ENV" python -m pip install ninja
"$CONDA" run -n "$CONDA_ENV" python build.py \
  --config Release \
  --cmake_generator Ninja \
  --no_telemetry \
  --ort_home "$RUNTIME_ROOT/oga-ort-home" \
  --skip_tests \
  --skip_examples \
  --parallel \
  --build_dir "$RUNTIME_ROOT/build-oga" \
  --cmake_extra_defines ENABLE_TESTS=OFF

```

### 4. Assemble the packaged runtime directory

The ORT, HIP EP, and OGA builds now exist in separate build/install
directories. The commands below assemble their required outputs into one
portable directory containing `bin/`, `lib/`, `etc/`, and `wheels/`. This is
file packaging only; no additional compilation happens here:

```bash
export HIPEP_ROOT="$RUNTIME_ROOT/linux-gpu-test-package"
rm -rf "$HIPEP_ROOT"
mkdir -p "$HIPEP_ROOT"
cp -a "$RUNTIME_ROOT/hip-install"/. "$HIPEP_ROOT"/
mkdir -p "$HIPEP_ROOT/bin" "$HIPEP_ROOT/lib" "$HIPEP_ROOT/etc" "$HIPEP_ROOT/wheels"

cp "$RUNTIME_ROOT/build-oga/Release/benchmark/c/model_benchmark" \
  "$HIPEP_ROOT/bin/"
cp -a "$RUNTIME_ROOT"/build-oga/Release/libonnxruntime-genai.so* \
  "$HIPEP_ROOT/lib/"
cp -a "$RUNTIME_ROOT"/ort-install/lib/libonnxruntime.so* \
  "$HIPEP_ROOT/lib/"
cp "$HIPEP_ROOT/lib/libhipgpu.so" "$HIPEP_ROOT/lib/libamdgpu-ep.so"
cp "$RUNTIME_ROOT/hip-ep/etc/morphizen_config.json" "$HIPEP_ROOT/etc/"
cp "$RUNTIME_ROOT"/ort-install/wheels/onnxruntime-*.whl \
  "$HIPEP_ROOT/wheels/"
cp "$RUNTIME_ROOT"/build-oga/Release/wheel/onnxruntime_genai-*.whl \
  "$HIPEP_ROOT/wheels/"
chmod +x "$HIPEP_ROOT/bin"/*
```

The resulting `HIPEP_ROOT` should contain `bin/`, `lib/`, `etc/`, and
`wheels/`. Install its two wheels into the fresh Conda environment:

```bash
"$CONDA" run -n "$CONDA_ENV" python -m pip install \
  "$HIPEP_ROOT"/wheels/onnxruntime-*.whl \
  "$HIPEP_ROOT"/wheels/onnxruntime_genai-*.whl \
  onnx
"$CONDA" run -n "$CONDA_ENV" python -m pip check

eval "$("$CONDA" shell.bash hook)"
conda activate "$CONDA_ENV"
```

## 1. Install DFlash dependencies

Section 0 creates the required isolated Conda environment, builds the matching
ORT/HIP EP/OGA bundle, installs its wheels, and activates the environment. The
commands below assume that setup has completed successfully; do not substitute
an unrelated HIP EP package or OGA wheel.

### Install DFlash Dependencies from Upstream DFlash Repo

Install the DFlash checkout. The DFlash draft runs on CPU; the target
model runs through OGA/HIP EP.
This flow pins the original DFlash code at `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`,
before the upstream DFlash2 generation path was added.

```bash
git clone https://github.com/z-lab/dflash.git dflash-upstream
cd dflash-upstream
git checkout 94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756
python -m pip install -e .
```

Install a CPU PyTorch wheel compatible with this Python environment:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Install the remaining Python dependencies used by the benchmark and profiling
scripts:

```bash
python -m pip install "transformers==5.15.0" accelerate typing-extensions datasets tqdm
```

## 2. Build the Modified OGA Target for DFlash

The target adapter selects its execution path from `text.onnx`:

- A model with `state_window` attributes on its supported recurrent operators
uses KV/position cropping and recurrent-state slot promotion after partial
acceptance. It does not replay the accepted prefix.
- A model without `state_window` uses the existing rewind-and-replay path.

The no-replay windowed path requires a state-window export with matching
past/present state names and an OGA runtime containing
`Generator.crop_to_accepted()`. If that method is unavailable, the adapter
falls back to rewind/replay.
The HIP EP package must be built from the corresponding state-window HIP EP
change; this DFlash directory does not contain the HIP EP compiler or kernel
implementation.

To produce the original windowed OGA source package from the non-windowed
Qwen3.6-27B export, follow
`[OGA_WINDOWED_STATE_EXPORT.md](OGA_WINDOWED_STATE_EXPORT.md)`.

The first DFlash-specific step is to derive an OGA target model that exposes the hidden-state taps and full-block logits needed by the Python DFlash loop. The
source OGA model is left unchanged.

```bash
export SOURCE_MODEL=/absolute/path/to/Qwen3.6-27B_gpuep_1.0_gpu_128
export WINDOWED_SOURCE_MODEL=/absolute/path/to/Qwen3.6-27B_gpuep_1.0_gpu_128_state_window8_hipep

: "${HIPEP_ROOT:?Set HIPEP_ROOT to the packaged HIP EP distribution}"
require_path() {
  test -e "$1" || {
    echo "Missing required path: $1" >&2
    return 1
  }
}
require_path "$HIPEP_ROOT" &&
require_path "$SOURCE_MODEL/text.onnx" &&
require_path "$SOURCE_MODEL/genai_config.json" &&
require_path "$WINDOWED_SOURCE_MODEL/text.onnx" &&
require_path "$WINDOWED_SOURCE_MODEL/genai_config.json" &&
echo "Model and HIP EP paths: PASS"
unset -f require_path

python tools/dflash-hipep/build_onnx_target.py \
  --source "$SOURCE_MODEL" \
  --output tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --target-layer-spec z-lab/Qwen3.6-27B-DFlash \
  --force

```

The command above derives the target from the ordinary, non-windowed Qwen
export. Derive the windowed target separately from the matching windowed OGA
export; do not reuse the non-windowed derived directory:

```bash
python tools/dflash-hipep/build_onnx_target.py \
  --source "$WINDOWED_SOURCE_MODEL" \
  --output tools/dflash-hipep/models/Qwen3.6-27B-hidden-windowed \
  --target-layer-spec z-lab/Qwen3.6-27B-DFlash \
  --force
```

### Test the Modified OGA Model

Run the fast adapter contract tests first. These do not load the full model.

```bash
python -m unittest tools/dflash-hipep/tests/test_oga_target_contract.py
```

The adapter contract tests cover both the legacy rewind/replay path and the
windowed crop/no-replay path without loading the full model.

```bash
python tools/dflash-hipep/tests/probe_oga_native_snapshot_restore.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --hipep-root "$HIPEP_ROOT"
```

Run the derived-target integration check for a windowed model. This validates
that the target loads through HIP EP, exposes the required hidden-state taps
and logits, and performs one crop with zero replay appends after partial
acceptance.

```bash
python tools/dflash-hipep/tests/validate_oga_target_integration.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden-windowed \
  --hipep-root "$HIPEP_ROOT" \
  --expect-mode window-crop
```

Run the integration check for the ordinary non-windowed target as well. This must select the rewind/replay compatibility path:

```bash
python tools/dflash-hipep/tests/validate_oga_target_integration.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --hipep-root "$HIPEP_ROOT" \
  --expect-mode rewind-replay
```

For the Qwen3.6-27B windowed export, verify the actual slot promotion across
all exposed recurrent and convolution state tensors:

```bash
python tools/dflash-hipep/tests/probe_oga_window_crop_slots.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden-windowed \
  --hipep-root "$HIPEP_ROOT"
```

The probe should report exact promotion for every destination tensor and no
changes to non-destination slots.

The tests above verify that the integration is wired correctly: the target
loads, the required hidden-state taps are exposed, the intended rollback mode
is selected, snapshots restore exactly, and window slots are promoted
bit-for-bit.

## 3. Run Saved Wall-Profile Benchmarks and Accuracy Checks

We compare vanilla OGA target-only throughput (vanilla decoding), adapter block-size-1 throughput, and DFlash at the requested block sizes. It also saves a JSON summary with wall-time breakdowns, acceptance statistics, token parity, and boxed-answer parity. This is wall profiling from Python timers, not a `rocprof` trace.

```bash
RESULTS=tools/dflash-hipep/results/patched_oga_27b_gsm8k_20x512
mkdir -p "$RESULTS"

python tools/dflash-hipep/profile_oga_dflash.py \
  --target tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --draft-model z-lab/Qwen3.6-27B-DFlash \
  --tokenizer "$SOURCE_MODEL" \
  --max-samples 20 \
  --max-new-tokens 512 \
  --block-sizes 4 5 7 16 \
  --output "$RESULTS/profile_27b_gsm8k_20x512_b4_b5_b7_b16.json" \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu \
  2>&1 | tee "$RESULTS/profile_27b_gsm8k_20x512_b4_b5_b7_b16.log"
```

Results that show using unwindowed path, bs7 has highest relative speedup. We then use bs7 for analyzing further improvement from windowed path down below.


| **Block size** | **Speedup vs clean OGA** | **Dflash boxed answer vs bs1 boxed answer** |
| -------------- | ------------------------ | ------------------------------------------- |
| 4              | 1.62×                    | 17/17                                       |
| 5              | 1.69×                    | 17/17                                       |
| 7              | 1.93×                    | 17/17                                       |
| 16             | 1.02×                    | 17/17                                       |


For the final Qwen3.6-27B GSM8K comparison, run 100 samples for the windowed
no-replay path and the original non-windowed rewind/replay path. Use separate
output directories so the results cannot be mixed:

```bash
for mode in windowed unwindowed; do
  mkdir -p "tools/dflash-hipep/results/qwen36_27b_gsm8k_100_${mode}"
done

python tools/dflash-hipep/profile_oga_dflash.py \
  --target tools/dflash-hipep/models/Qwen3.6-27B-hidden-windowed \
  --draft-model z-lab/Qwen3.6-27B-DFlash \
  --tokenizer "$SOURCE_MODEL" \
  --max-samples 100 --max-new-tokens 512 --block-sizes 7 \
  --output tools/dflash-hipep/results/qwen36_27b_gsm8k_100_windowed/profile.json \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu

python tools/dflash-hipep/profile_oga_dflash.py \
  --target tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --draft-model z-lab/Qwen3.6-27B-DFlash \
  --tokenizer "$SOURCE_MODEL" \
  --max-samples 100 --max-new-tokens 512 --block-sizes 7 \
  --output tools/dflash-hipep/results/qwen36_27b_gsm8k_100_unwindowed/profile.json \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu

python tools/dflash-hipep/gsm8k_eval.py \
  --results-file tools/dflash-hipep/results/qwen36_27b_gsm8k_100_windowed/profile.json \
  --output-file tools/dflash-hipep/results/qwen36_27b_gsm8k_100_windowed/accuracy.json \
  --block-size 7

python tools/dflash-hipep/gsm8k_eval.py \
  --results-file tools/dflash-hipep/results/qwen36_27b_gsm8k_100_unwindowed/profile.json \
  --output-file tools/dflash-hipep/results/qwen36_27b_gsm8k_100_unwindowed/accuracy.json \
  --block-size 7
```

Token divergence is not inherent to DFlash: the original PyTorch implementation preserves token parity between block-size-1 and wide speculative decoding when the model is loaded in FP32. In this exported OGA/HIP EP path, however, wide (`M > 1`) and sequential (`M = 1`) execution can use different provider kernels and floating-point reduction orders at the model's configured precision. The resulting small logit differences can eventually select a different token near a decision boundary. The windowed path additionally retains recurrent state from the wide verification pass, whereas rewind/replay regenerates accepted state sequentially. Because divergence has been observed with both rollback modes, it is not by itself evidence of an incorrect crop implementation. Use the GSM8K evaluation below for the end-to-end comparison, with boxed-answer correctness as the accuracy criterion.

These should be the results seen:


| **Metric**                     | **Windowed crop/no-replay** | **Non-windowed replay** |
| ------------------------------ | --------------------------- | ----------------------- |
| Clean OGA                      | 12.28 tok/s                 | 13.23 tok/s             |
| DFlash B7                      | 31.40 tok/s                 | 26.30 tok/s             |
| Speedup vs clean               | 2.56×                       | 1.99×                   |
| Adapter ground-truth GSM8K     | 78/100                      | 76/100                  |
| DFlash ground-truth GSM8K      | 75/100                      | 79/100                  |
| DFlash vs adapter boxed parity | 80/80                       | 81/81                   |
