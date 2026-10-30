# DFlash on HIP EP

## 1. Environment Setup

This setup uses a clean Conda environment named `dflash_hipep_latest_clean`.

The HIP EP runtime package used here is the `linux-gpu-test-package` prebuilt package downloaded on 2026-10-19.

Create the environment and install the HIP EP package wheels:

```bash
conda create -n dflash_hipep_latest_clean python=3.14 pip -y
conda activate dflash_hipep_latest_clean

export HIPEP_ROOT=<path to>/hipep_latest_prebuilt/linux-gpu-test-package
chmod +x "$HIPEP_ROOT"/bin/*

python -m pip install --upgrade pip
python -m pip install \
  "$HIPEP_ROOT"/wheels/onnxruntime-*.whl \
  "$HIPEP_ROOT"/wheels/onnxruntime_genai-*.whl \
  onnx
```


### Get Upstream OGA and Apply the Snapshot/Rewind Patch

The HIP EP prebuilt package provides the ONNX Runtime and baseline OGA wheels.
Its OGA 0.17 wheel exposes `Generator.snapshot_state()` and
`Generator.rewind_to()`, but those APIs do not restore Qwen3.6 hybrid recurrent
state through the multimodal pipeline. DFlash partial rejection requires that
restore behavior, so build a patched OGA wheel from upstream ONNX Runtime GenAI
source at `5c19002edce50beaa3a5c7eeefd8fe5269ee8d19` and install the resulting
`0.18.0-dev` wheel on top of the HIP EP package.

```bash
cd tools/dflash-hipep
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
git checkout 5c19002edce50beaa3a5c7eeefd8fe5269ee8d19

git apply ../patches/oga-main-5c19002-amdgpu-multimodal-snapshot.patch
git diff --stat
```

Build and install the patched OGA wheel without replacing the HIP EP package's
ONNX Runtime wheel:

```bash
export OGA_BUILD=./hip-ep/tools/dflash-hipep/build-oga-patched

python -m pip install -r requirements-dev.txt
python -m pip install ninja

python build.py \
  --config Release \
  --skip_tests \
  --skip_examples \
  --cmake_generator Ninja \
  --build_dir "$OGA_BUILD"

python -m pip install --force-reinstall --no-deps \
  "$OGA_BUILD"/Release/wheel/onnxruntime_genai-*.whl
```



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

The first DFlash-specific step is to derive an OGA target model that exposes the hidden-state taps and full-block logits needed by the Python DFlash loop. The
source OGA model is left unchanged.

```bash
python tools/dflash-hipep/build_onnx_target.py \
  --source <path-to-orig-gpuep-oga-model> \
  --output tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --target-layer-spec z-lab/Qwen3.6-27B-DFlash \
  --force

python tools/dflash-hipep/build_onnx_target.py \
  --source <path-to-Qwen3.6-35B-A3B-gpuep-OGA-model> \
  --output tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden \
  --target-layer-spec z-lab/Qwen3.6-35B-A3B-DFlash \
  --force
```



### Test the Modified OGA Model

Run the fast adapter contract tests first. These do not load the full model.

```bash
python -m unittest tools/dflash-hipep/tests/test_oga_target_contract.py
```

Validate that OGA native snapshot/rewind restores Qwen3.6 hybrid recurrent
state exactly. This must pass before running DFlash parity or benchmarks.

```bash
python tools/dflash-hipep/tests/probe_oga_native_snapshot_restore.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --hipep-root "$HIPEP_ROOT"

python tools/dflash-hipep/tests/probe_oga_native_snapshot_restore.py \
  --model tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden \
  --hipep-root "$HIPEP_ROOT"
```

Run the derived-target integration check for each model. This validates that the
derived OGA target loads through HIP EP, exposes the required hidden-state taps
and logits, and supports the snapshot/rewind path used by the DFlash loop.

```bash
python tools/dflash-hipep/tests/validate_oga_target_integration.py \
  --model tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --hipep-root "$HIPEP_ROOT"

python tools/dflash-hipep/tests/validate_oga_target_integration.py \
  --model tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden \
  --hipep-root "$HIPEP_ROOT"
```

Validate exact output parity between the untouched OGA target, the derived
target running at block size 1, and the DFlash speculative path for each model.

```bash
python tools/dflash-hipep/tests/validate_oga_dflash.py \
  --source-target /<path to orig model>/Qwen3.6-27B_gpuep_1.0_gpu_128 \
  --target tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --draft-model z-lab/Qwen3.6-27B-DFlash \
  --block-size 16 \
  --max-new-tokens 128 \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu

python tools/dflash-hipep/tests/validate_oga_dflash.py \
  --source-target /<path to orig model>/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128 \
  --target tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden \
  --draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --block-size 16 \
  --max-new-tokens 128 \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu
```

Each parity case should report `derived_vs_original=True`,
`speculative_vs_derived=True`, and `speculative_vs_original=True`. This proves
that the derived OGA target matches the untouched OGA target and that the
OGA-backed DFlash speculative loop produces the same greedy token stream.

## 3. Run Saved Wall-Profile Benchmarks and Accuracy Checks

We compare vanilla OGA target-only throughput (vanilla decoding), adapter block-size-1 throughput, and DFlash at the requested block sizes. It also saves a JSON summary with wall-time breakdowns, acceptance statistics, token parity, and boxed-answer parity. This is wall profiling from Python timers, not a `rocprof` trace.

```bash
RESULTS=tools/dflash-hipep/results/patched_oga_27b_gsm8k_20x512
mkdir -p "$RESULTS"

python tools/dflash-hipep/profile_oga_dflash.py \
  --target tools/dflash-hipep/models/Qwen3.6-27B-hidden \
  --draft-model z-lab/Qwen3.6-27B-DFlash \
  --tokenizer <path-to-Qwen3.6-27B-gpuep-OGA-model> \
  --max-samples 20 \
  --max-new-tokens 512 \
  --block-sizes 4 5 7 16 \
  --output "$RESULTS/profile_27b_gsm8k_20x512_b4_b5_b7_b16.json" \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu \
  2>&1 | tee "$RESULTS/profile_27b_gsm8k_20x512_b4_b5_b7_b16.log"
```

For 35B-A3B:

```bash
RESULTS=tools/dflash-hipep/results/patched_oga_35b_gsm8k_20x512
mkdir -p "$RESULTS"

python tools/dflash-hipep/profile_oga_dflash.py \
  --target tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden \
  --draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --tokenizer <path-to-Qwen3.6-35B-A3B-gpuep-OGA-model> \
  --max-samples 20 \
  --max-new-tokens 512 \
  --block-sizes 4 5 7 16 \
  --output "$RESULTS/profile_35b_gsm8k_20x512_b4_b5_b7_b16.json" \
  --hipep-root "$HIPEP_ROOT" \
  --draft-device cpu \
  2>&1 | tee "$RESULTS/profile_35b_gsm8k_20x512_b4_b5_b7_b16.log"
```

For a native `rocprofv3` trace, run one sample and one block size while enabling
ROCTX ranges:

```bash
export RESULTS=tools/dflash-hipep/results/rocprof_27b_b7_sample0
mkdir -p "$RESULTS"
rocprofv3 --hip-trace --marker-trace --output-directory "$RESULTS" --output-format csv -- python tools/dflash-hipep/profile_oga_dflash.py --target tools/dflash-hipep/models/Qwen3.6-27B-hidden --draft-model z-lab/Qwen3.6-27B-DFlash --tokenizer <path-to-Qwen3.6-27B-gpuep-OGA-model> --max-samples 1 --sample-indices 0 --max-new-tokens 512 --block-sizes 7 --skip-clean-baseline --skip-adapter-baseline --roctx-ranges --output "$RESULTS/profile_27b_gsm8k_sample0_b7_roctx.json" --hipep-root "$HIPEP_ROOT" --draft-device cpu 2>&1 | tee "$RESULTS/profile_27b_gsm8k_sample0_b7_roctx.log"
```

For Qwen3.6-35B-A3B:

```bash
export RESULTS=tools/dflash-hipep/results/rocprof_35b_b7_sample0
mkdir -p "$RESULTS"
rocprofv3 --hip-trace --marker-trace --output-directory "$RESULTS" --output-format csv -- python tools/dflash-hipep/profile_oga_dflash.py --target tools/dflash-hipep/models/Qwen3.6-35B-A3B-hidden --draft-model z-lab/Qwen3.6-35B-A3B-DFlash --tokenizer <path-to-Qwen3.6-35B-A3B-gpuep-OGA-model> --max-samples 1 --sample-indices 0 --max-new-tokens 512 --block-sizes 7 --skip-clean-baseline --skip-adapter-baseline --roctx-ranges --output "$RESULTS/profile_35b_gsm8k_sample0_b7_roctx.json" --hipep-root "$HIPEP_ROOT" --draft-device cpu 2>&1 | tee "$RESULTS/profile_35b_gsm8k_sample0_b7_roctx.log"
```

