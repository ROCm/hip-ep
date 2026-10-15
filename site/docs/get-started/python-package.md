---
title: Run models from Python
description: Install the four wheels bundled with HIP EP v0.5.1, select AMDGPU in an existing model directory, and run the Python text or vision-language benchmark.
---

The v0.5.1 Python archive contains four local wheels for running HIP EP through
ONNX Runtime and OGA, together with the benchmark scripts used below. Installing
the wheels places the EP's native files where the packaged runtime expects them;
no Visual Studio installation or manual DLL copy is part of this procedure.
The four project wheels are local, but pip may still contact its configured
package index for their declared Python dependencies and for the benchmark
helpers installed later.

This guide covers Strix Halo, Strix Point, Krackan Point, and GPT2 on Windows.
It assumes that the graphics driver is already installed. The commands use
PowerShell syntax, but HIP EP itself does not depend on PowerShell.
The v0.5.1 procedure uses `GPT2` as its fourth target label. Because that is not
a retail adapter name, confirm the matching product with the release owner
before using that target.

Before starting, obtain an OGA-ready ONNX model directory. It must contain
`genai_config.json` and the model and tokenizer files referenced by that
configuration. A base-model repository by itself is not enough. The
[Model Matrix]({{ '/docs/models/' | relative_url }}) identifies validated model
names and configurations, but it does not distribute the ready-to-run directories.

## 1. Prerequisites

| Item | Requirement |
|---|---|
| OS | Windows (amd64) |
| Shell | Windows PowerShell. The commands below use PowerShell syntax; HIP EP itself does not require a particular shell |
| GPU | Strix Halo, Strix Point, Krackan Point, or GPT2, with a current graphics driver. The driver supplies `amdhip64_7.dll`; the wheels supply the EP and its other packaged native dependencies |
| Python | 3.14 (`cp314`), matching the wheel ABI tag |
| Model | An existing OGA-ready ONNX directory containing `genai_config.json` and the files it references |
| Network | Required to download the package and model. Pip also needs access to its configured package index unless the wheels' Python dependencies and the benchmark helpers are already cached |

## 2. Download the Python package

Download
[`hipep_0.5.1_windows_python.zip`](https://github.com/ROCm/hip-ep/releases/download/v0.5.1/hipep_0.5.1_windows_python.zip).
The published archive contains four benchmark scripts and four wheels. It does
not contain a source distribution or any models.

The four wheels have distinct roles:

| Wheel | Role |
|---|---|
| `onnxruntime_directml-*.whl` | ONNX Runtime with Plugin EP support |
| `onnxruntime_ep_amdgpu-*.whl` | AMDGPU umbrella EP used by OGA to load the selected backend |
| `onnxruntime_ep_hip-*.whl` | HIP EP plugin, compiler, custom kernels, hipBLASLt data, and packaged JIT/CRT support; its native payload shares the AMDGPU package directory |
| `onnxruntime_genai_directml-*.whl` | OGA's tokenizer, KV-cache, and generation runtime |

Use a model directory you already have. Its only required edit is the
`genai_config.json` change in section 4.

## 3. Install

Extract the archive and install all four bundled wheels into a fresh Python 3.14
virtual environment. Keep that environment activated for the rest of the guide.

```powershell
mkdir hipep-wheels
Expand-Archive -Path .\hipep_0.5.1_windows_python.zip -DestinationPath .\hipep-wheels -Force

py -3.14 --version   # should print 3.14.x
# If that command fails, install Python 3.14, then run the version check again:
# winget install --id Python.Python.3.14 -e

py -3.14 -m venv .venv
.venv\Scripts\activate
python --version   # follows the virtual environment; should print 3.14.x

cd hipep-wheels
pip install (Get-ChildItem .\wheels\*.whl).FullName  # bash: pip install wheels/*.whl
cd ..
```

## 4. Set the EP in `genai_config.json`

Put the existing model directory beside `hipep-wheels`. Text models use
`benchmark_e2e.py`; vision-language models use `vlm_benchmark.py` and require a
local image. Both run through `run_onnx.py --benchmark`.

Before the first run, open `<model-directory>\genai_config.json`. Under `model`
→ `decoder` → `session_options`, replace the existing `provider_options` block
with:

```json
"provider_options": [
  {
    "AMDGPU": {
      "profile": "hip"
    }
  }
]
```

Leave the rest of the file unchanged. OGA treats the model directory as the
runtime configuration, so both benchmark scripts read the provider from this
file rather than from a separate HIP EP command-line flag.

## 5. Run

The commands below assume the virtual environment from section 3 is active and
the model configuration from section 4 is complete. Replace the model and image
placeholders with local paths.

### LLM benchmark

Install the Python modules used by the text benchmark, then run it through the
environment-preparing launcher:

```powershell
pip install psutil pandas tqdm

python hipep-wheels\run_onnx.py --benchmark hipep-wheels\benchmark_e2e.py -i <model-directory> -l 128 -g 128 -w 1
```

| Flag | Purpose |
|---|---|
| `--benchmark <script>` | Run a benchmark script after preparing the package environment |
| `-i <dir>` | OGA-ready model directory |
| `-l <n>` | Prompt length |
| `-g <n>` | Tokens to generate |
| `-w <n>` | Warm-up runs |

Example output; absolute numbers vary with the machine, driver, power profile,
and model:

```text
Args: batch_size = 1, prompt_length = 128, tokens = 128, max_length = 256
100%|##########| 5/5 [...]   (warmup)
100%|##########| 10/10 [...]   (benchmark)
Average Tokenization Latency (per token): 0.00196 ms
Average Tokenization Throughput (per token): 509988.89 tps
Average Prompt Processing Latency (per token): 2.2055 ms
Average Prompt Processing Throughput (per token): 453.40 tps
Average Token Generation Latency (per token): 23.677 ms
Average Token Generation Throughput (per token): 42.235 tps
Average Sampling Latency (per token): 0.1161 ms
Average Sampling Throughput (per token): 8613.26 tps
Average Wall Clock Time: 3.1839 s
Average Wall Clock Throughput: 81.975 tps
Results saved in genai_e2e!
```

### VLM benchmark

`vlm_benchmark.py` reports image-preprocessing time, time to first token, and
token-generation rate. Replace `<model-directory>` and `<image>` with local
paths.

```powershell
pip install psutil pillow numpy

python hipep-wheels\run_onnx.py --benchmark hipep-wheels\vlm_benchmark.py `
  --model_path <model-directory> `
  --image_path <image> `
  --max_tokens 128 --max_length 4096 `
  --num_iterations 5 --warmup_iterations 1 `
  --output_json vlm-results.json --verbose
```

| Flag | Purpose |
|---|---|
| `--benchmark <script>` | Run a benchmark script after preparing the package environment |
| `--model_path <dir>` | OGA-ready model directory |
| `--image_path <path>` | Input image |
| `--max_tokens <n>` | Maximum tokens to generate |
| `--max_length <n>` | Maximum sequence length |
| `--num_iterations <n>` | Benchmark runs |
| `--warmup_iterations <n>` | Warm-up runs |
| `--output_json <path>` | Results file |
| `--verbose` | Verbose output |

Example output; absolute numbers vary by machine:

```text
Loading model...
Model loaded: qwen3_5_moe
Device: AMDGPU
Original image size: 225x225
Benchmark image size: 225x225
Model type: 'qwen3_5_moe' — using chat template from tokenizer_config.json
  Loaded chat template from: Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml\chat_template.jinja

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this image in detail.<|im_end|>
<|im_start|>assistant
<think>
...
Running 1 warmup iteration(s)...
The user wants a detailed description of the provided image.

......

Running 5 benchmark iteration(s)...
  Iteration 1/5...   Token breakdown: total=82 (text=19, image=63)
  Iteration 1: Preprocess: 5.3ms, TTFT: 338.8ms, Tokens: 128

  Generated output:
  --------------------------------------------------
  The user wants a detailed description of the provided image.

......

  --------------------------------------------------

  Iteration 2/5...   Iteration 2: Preprocess: 6.0ms, TTFT: 332.2ms, Tokens: 128
  Iteration 3/5...   Iteration 3: Preprocess: 6.1ms, TTFT: 332.5ms, Tokens: 128
  Iteration 4/5...   Iteration 4: Preprocess: 7.3ms, TTFT: 331.2ms, Tokens: 128
  Iteration 5/5...   Iteration 5: Preprocess: 7.2ms, TTFT: 334.2ms, Tokens: 128

============================================================
VLM BENCHMARK RESULTS
============================================================
Model Type: qwen3_5_moe
Image Size: 225x225
Number of runs: 5
------------------------------------------------------------

Token Breakdown:
  Total Prompt Tokens: 82
  Text Tokens:         19
  Image Tokens:        63

Preprocessing Time (image processing):
  Average:        6.38 ms
  Std Dev:        0.85 ms
  P50:            6.05 ms
  P90:            7.33 ms
  P99:            7.33 ms
  Min:            5.32 ms
  Max:            7.33 ms

Time To First Token (TTFT):
  Average:      333.78 ms
  Std Dev:        3.02 ms
  P50:          332.51 ms
  P90:          338.82 ms
  P99:          338.82 ms
  Min:          331.15 ms
  Max:          338.82 ms

Prefill Throughput:
  Average:      245.69 tokens/sec
  P50:          246.61 tokens/sec

Token Generation (excluding first token):
  Average:       17.34 ms/token
  TPS:           57.66 tokens/sec
  Std Dev:        0.50 ms
  P50:           17.29 ms/token (57.84 TPS)
  P90:           17.69 ms/token (56.54 TPS)
  Min:           16.70 ms/token
  Max:           24.38 ms/token

Tokens Generated:
  Average per run: 128.0
  Total:           640

End-to-End Time (Preprocessing + TTFT + Token Gen):
  Average:     2542.73 ms
  P50:         2534.96 ms

Peak Memory Usage:
      685.59 MB (0.67 GB)
============================================================

Results exported to: vlm-results.json
```

`Device: AMDGPU` confirms that the AMDGPU provider is active for this run.

## 6. Troubleshooting

`run_onnx.py --benchmark` imports the AMDGPU package, which adds its native DLL
directory to `PATH`; the wrapper also sets `AMDGPU_EP_PATH` and points `LIB` at
that directory before starting the selected benchmark script. Use the wrapper
instead of launching either benchmark script directly.

| Symptom | Cause / Fix |
|---|---|
| `... is not a supported wheel on this platform` | Python is not 3.14. Recreate the venv with `py -3.14 -m venv .venv` |
| `lld-link: could not open 'amdhip64.lib'` | Use `run_onnx.py --benchmark`, not the benchmark script directly. If the wrapper still reports this error after all four bundled wheels were installed together in a fresh environment, the package is missing an import library required by that compilation path; record the wheel filenames and report the package issue |
| `lld-link: could not open 'msvcrt.lib'` | Use `run_onnx.py --benchmark`, which points `LIB` at the directory containing the packaged CRT import libraries. If it still fails, reinstall all four wheels together in a fresh environment |
| `Failed to load ... amdhip64_7.dll` | Install or update the AMD graphics driver. This DLL comes from the driver rather than from a separate ROCm wheel in the archive |
| EP not selected (falls back to CPU) | Confirm that `genai_config.json` contains `[{"AMDGPU":{"profile":"hip"}}]` and run through the wrapper. `Device: AMDGPU` is the direct confirmation in the VLM output; the text benchmark output does not expose an equivalent device line |
