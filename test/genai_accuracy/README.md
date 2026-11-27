<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# GenAI Accuracy Gate

`run_genai_accuracy.py` runs an end-to-end GenAI accuracy test on the AMDGPU
(HIP) EP from a `gpu-test-package`. It uses the
[hipdnn-accuracy-test](https://github.com/AMDmoore/hipdnn-accuracy-test)
framework and is the GitHub Actions port of the HIP EP GenAI Jenkins accuracy
flow (`TEST_MODE=accuracy`, `GENAI_ACCURACY_TESTS=PPL`, `LLM_RUN_EPS=hip`).

The `Windows GPU Test` workflow (`.github/workflows/windows-gpu-test.yml`) runs
it on every PR, merge group, and push.

## What runs

The workflow runs these models in order, each through the same steps. The test
follows the modelzoo profile, as in the Jenkins flow:

| Model | Test | Source |
|-------|------|--------|
| `Qwen3.8-Flash-Next` | `PPL_VLM` | Hugging Face `amd/Qwen3.8-Flash-Next_gpuep_1.0_gpu_128`, pinned revision |
| `Qwen3.8-27B` | `PPL_VLM` | model share |
| `Gemma-4-E4B-it` | `PPL_VLM` | model share |
| `gpt-oss-20b` | `PPL` | model share |
| `Llama-3.1-8B` | `PPL` | model share |

- `PPL_VLM` (VLM profile): conditional perplexity of `lmms-lab/flickr30k` test
  captions given the image. Parameters come from the framework's
  `test_config_vlm.json`.
- `PPL` (LLM profile): wikitext2 perplexity with a 512-token window over 2% of the
  corpus (`nsamples=0.02`); the other parameters, such as `stride`, come from the
  framework's `test_config.json`.

| Item | Value |
|------|-------|
| EP | `AMDGPU` with `profile: hip` in every session, `session.disable_cpu_ep_fallback=1`, `search.chunk_size=512` (`Gemma-4-E4B-it` sets no chunk size and uses `enable_graph_capture=0`) |
| Framework | hipdnn-accuracy-test at a pinned commit |
| Python | 3.14 venv with the package's `onnxruntime_directml` and `onnxruntime_genai_directml` wheels |

`session.disable_cpu_ep_fallback=1` makes session creation fail if any node
would fall back to the CPU EP. A passing run therefore measured the GPU path.

## Pass criteria

1. `run_accuracy.py` exits 0 and reports a `Perplexity:` value for every model
   that ran. Otherwise the `Run GenAI accuracy` step fails. A skipped model (see
   below) does not fail the step.
2. On pull requests and merge groups, each model's perplexity may rise by at
   most 2% over its `<model>/result.json` in the `genai-accuracy-results`
   artifact of the latest successful `Windows Build` push run on `main` that
   has one. Lower perplexity is better. A model's baseline is used only when its
   model revision, framework commit, and test parameters match this run. Pushes
   report the delta without gating. A model with no comparable baseline is not
   gated.

Results appear in the job summary and, on PRs, in one comment marked
`<!-- morphizen-genai-accuracy-results -->` with a section per model. The
`genai-accuracy-results` artifact has one subdirectory per model
(`<model>/result.json`, the framework logs, and the generated
`genai_config_AMDGPU.json`).

## Runner layout

`--cache-root` (`<runner.workspace>/genai-accuracy` in CI) persists across runs:

```text
hipdnn-accuracy-test/   framework checkout at the pinned commit
genai-accuracy-venv/    venv, rebuilt when the Python or pinned packages change
hf-home/                Hugging Face cache (datasets)
genai-models/<model>    link to the model in the model store
```

The models live in the machine-wide model store `--models-root` (default
`$GENAI_MODELS_ROOT`, or `C:\models` on Windows) as `<models-root>/<model>`, so
every runner workspace shares one copy. `prepare` sets up the framework, venv and
wheels once, then for each model:

1. Uses the model in the store when `<models-root>/<model>/genai_config.json`
   exists and no `.download_incomplete` marker is left by an interrupted copy or
   download. A model placed there by hand is used as is.
2. Otherwise copies it from the model share `--source-root` (default
   `$GENAI_MODELS_SOURCE`, or `R:/` on Windows) into `<models-root>/<model>`
   with `robocopy`, which resumes a partial copy.
3. Otherwise, for a model with a Hugging Face source, downloads the pinned
   revision.
4. Links `<cache-root>/genai-models/<model>` to the store directory: a symlink,
   or a directory junction when the account cannot create symlinks.

One model failing to copy or download does not stop the others; `prepare` then
exits non-zero.

Each run hard-links the model into its work dir, because `run_accuracy.py`
overwrites `genai_config.json` in the model directory. The work dir must be on
the same volume as the model store. The five models take roughly 150 GB or more
in the store.

### Model share layout

The share is mapped as `R:/` on the runner.
A drive mapped in an interactive session is not visible to the runner
service. When `R:` is missing, the CI prepare step emits a warning, and
`prepare` logs why each share lookup failed (share not accessible, directory
missing, or no `genai_config.json`).

It holds one directory per model, named after the modelzoo alias. Each must
contain `genai_config.json`:

| Model | Directory under `R:/` |
|-------|----------------------|
| `Qwen3.8-27B` | `Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-dml-onnx` (16 GB) |
| `Gemma-4-E4B-it` | `gemma-4-E4B-it-awq-uint4-onnx-split-fp16-pruned_lm_head` (11 GB) |
| `gpt-oss-20b` | `gpt-oss-20b-webgpu-int4-rtn-block-32-pruned_lm_head` (12 GB) |
| `Llama-3.1-8B` | `Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml-pruned_lm_head` (4.7 GB) |

The directory names are the `MODELS[...]["source_dirs"]` entries in
`run_genai_accuracy.py`. These models have no pinned revision; the copy from the
share is used as is.

### Skipped models

A model with no Hugging Face source that is neither in the store nor on the
share is skipped, not failed: `prepare` logs `[SKIP]`, `run` writes a
`result.json` with `skipped: true`, the report shows `SKIPPED`, and the job does
not gate on it.

## Running locally

On a Windows Strix Halo machine with Python 3.14 and an extracted
`gpu-test-package`:

```bash
python test/genai_accuracy/run_genai_accuracy.py prepare \
  --model Qwen3.8-27B gpt-oss-20b \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --models-root D:/models --source-root R:/
python test/genai_accuracy/run_genai_accuracy.py run \
  --model gpt-oss-20b \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --work-dir D:/genai-accuracy-work --results-dir genai-accuracy-results
python test/genai_accuracy/run_genai_accuracy.py report \
  --current genai-accuracy-results/result.json \
  --baseline main/result.json \
  --summary-out summary.md --regressions-out regressions.txt
```

## Changing the pins

Changing `ACCURACY_REPO_COMMIT`, a model `revision`, a model's test parameters, or
the framework's `PPL`/`PPL_VLM` parameters makes the main baseline non-comparable
for the affected models. The first PR after such a change runs ungated for them.
Gating resumes once that change has a successful run on `main`.
