---
title: Run models with the binary package
description: Download the HIP EP v0.5.1 Windows test package, select AMDGPU in an existing model directory, and run its text or vision-language benchmark.
---

The Windows GPU test package is a portable HIP EP installation. It contains the
EP and OGA/ONNX Runtime binaries, the user-mode ROCm runtime, and the MSVC/UCRT
import libraries needed by the JIT linker. Extracting the archive is the only
installation step; it does not register system components.

This guide covers release **v0.5.1** on Strix Halo, Strix Point, Krackan Point,
and GPT2. It assumes that Windows and the graphics driver are already installed.
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
| GPU | Strix Halo, Strix Point, Krackan Point, or GPT2, with a current graphics driver. The archive supplies the user-mode HIP runtime DLLs; the installed driver supplies the kernel-mode GPU support |
| Python | 3.14 (`cp314`) for the VLM benchmark only. The wheels in `wheels/` use this ABI; `model_benchmark.exe` does not use Python |
| Model | An existing OGA-ready ONNX directory containing `genai_config.json` and the files it references |
| Network | Required to download the package and model. The VLM dependency install also uses pip's configured package index unless those packages are already cached |

## 2. Download and extract the test package

Download
[`hipep_0.5.1_windows.zip`](https://github.com/ROCm/hip-ep/releases/download/v0.5.1/hipep_0.5.1_windows.zip).
The archive contains the portable `bin/`, `lib/`, and `wheels/` directories.
Models are distributed separately.

`tar` on Windows 10 and later extracts `.zip` files:

```powershell
mkdir gpu-test-package
tar -xf hipep_0.5.1_windows.zip -C gpu-test-package
```

Use an existing model directory and edit only its `genai_config.json` before
running. Text models use `model_benchmark.exe`. Vision-language models use
`vlm_benchmark.py` and also require a local image.

```text
<working-directory>\
  hipep_0.5.1_windows.zip
  gpu-test-package\
    bin\
    lib\
    wheels\
  <model-directory>\
    genai_config.json
  <image>
```

## 3. Set the EP in `genai_config.json`

OGA reads execution-provider settings from each model directory. Before the
first run, open `<model-directory>\genai_config.json`. Under `model` → `decoder`
→ `session_options`, replace the existing `provider_options` block with:

```json
"provider_options": [
  {
    "AMDGPU": {
      "profile": "hip"
    }
  }
]
```

Leave the rest of the file unchanged. Both `model_benchmark.exe` and
`vlm_benchmark.py` read this block. This model-level setting is an OGA contract:
the benchmark command does not independently override the model's provider.

## 4. Run

The commands below assume section 3 is complete. Replace `<model-directory>`
with the directory already present in your working folder.

### LLM benchmark

Run `model_benchmark.exe` by its full path so it remains next to the DLLs in
`bin`.

```powershell
.\gpu-test-package\bin\model_benchmark.exe -i <model-directory> -l 128 -g 128 -r 5 -w 1 -b 1 -v
```

| Flag | Purpose |
|---|---|
| `-i <dir>` | OGA-ready model directory |
| `-l <n>` | Prompt length |
| `-g <n>` | Tokens to generate |
| `-r <n>` | Benchmark repetitions |
| `-w <n>` | Warm-up runs |
| `-b <n>` | Batch size |
| `-v` | Verbose output |

Example output:

```text
Running warmup iterations (1)...
[PROMPT BEGIN]
[OUTPUT BEGIN]
Running iterations (5)...
Batch size: 1, prompt tokens: 128, tokens to generate: 128
```

### VLM Benchmark (multimodal models)

`bin/vlm_benchmark.py` reports image-preprocessing time, time to first token,
and token-generation rate. The commands below create a Python 3.14 environment
and run the benchmark. Replace `<model-directory>` and `<image>` with local
paths.

```powershell
py -3.14 --version
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install (Get-Item gpu-test-package\wheels\*.whl)   # bash: pip install gpu-test-package/wheels/*.whl
pip install psutil pillow numpy

cd gpu-test-package\bin
python vlm_benchmark.py `
  --model_path ..\..\<model-directory> `
  --image_path ..\..\<image> `
  --max_tokens 128 --max_length 4096 `
  --num_iterations 5 --warmup_iterations 1 `
  --output_json ..\..\vlm-results.json --verbose
```

| Flag | Purpose |
|---|---|
| `--model_path <dir>` | OGA-ready model directory |
| `--image_path <path>` | Input image |
| `--max_tokens <n>` | Maximum tokens to generate |
| `--max_length <n>` | Maximum sequence length |
| `--num_iterations <n>` | Benchmark runs |
| `--warmup_iterations <n>` | Warm-up runs |
| `--output_json <path>` | Results file |
| `--verbose` | Verbose output |

In the output below, `Device: AMDGPU` is the benchmark's direct confirmation
that the AMDGPU provider is active:

```text
Loading model...
Device: AMDGPU
Prompt: <|im_start|>user
Running 1 warmup iteration(s)...
Running 5 benchmark iteration(s)...

============================================================
VLM BENCHMARK RESULTS
============================================================
```
