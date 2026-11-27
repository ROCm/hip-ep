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

The `genai_accuracy` job of the `Windows GPU Test` workflow
(`.github/workflows/windows-gpu-test.yml`) runs it on every PR, merge group,
and push, as a matrix with one leg per self-hosted GPU runner. It currently
runs on one runner:

| Runner label | Results artifact |
|--------------|------------------|
| `Strix` | `genai-accuracy-results-strix` |

Each runner is gated and reported on its own. To add a runner, add a matrix
entry with its label and a unique artifact name; the legs run in parallel.

## What runs

The workflow runs these models in order, each through the same steps. The test
follows the modelzoo profile, as in the Jenkins flow:

| Model | Test | Source |
|-------|------|--------|
| `Qwen3.8-Flash-Next` | `PPL_VLM` | Hugging Face `amd/Qwen3.8-Flash-Next_gpuep_1.0_gpu_128`, pinned revision |
| `gpt-oss-20b-webgpu-int4-rtn-block-32` | `PPL` | OGA/VLM benchmark model staged on the runner |
| `Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml` | `PPL` | OGA/VLM benchmark model staged on the runner |
| `gemma-4-E4B-it-fp16-int4-kquant-prune-logits` | `PPL_VLM` | OGA/VLM benchmark model staged on the runner |
| `Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-dml-onnx` | `PPL_VLM` | OGA/VLM benchmark model staged on the runner |

The staged models are the same ones the `Run OGA benchmark` and `Run VLM
benchmark` steps use: `<runner.workspace>/amdgpu-oga-models/<model>` or
`<runner.workspace>/amdgpu-vlm-models/<model>`, looked up by directory name.
They are used in place, not copied, and must be staged on every runner the job
runs on.

- `PPL_VLM` (VLM profile): conditional perplexity of `lmms-lab/flickr30k` test
  captions given the image. Parameters come from the framework's
  `test_config_vlm.json`.
- `PPL` (LLM profile): wikitext2 perplexity with a 512-token window over 2% of
  the corpus (`nsamples=0.02`); the other parameters, such as `stride`, come from
  the framework's `test_config.json`.

| Item | Value |
|------|-------|
| EP | `AMDGPU` with `profile: hip` in every session, `session.disable_cpu_ep_fallback=1`, `search.chunk_size=512` (`gemma-4-E4B-it-fp16-int4-kquant-prune-logits` sets no chunk size and uses `enable_graph_capture=0`; a staged model whose `genai_config.json` already selects `AMDGPU` keeps its provider options) |
| Framework | hipdnn-accuracy-test at a pinned commit |
| Python | 3.14 venv with the package's `onnxruntime_directml` and `onnxruntime_genai_directml` wheels |

`session.disable_cpu_ep_fallback=1` makes session creation fail if any node
would fall back to the CPU EP. A passing run therefore measured the GPU path.

## Pass criteria

1. `run_accuracy.py` exits 0 and reports a `Perplexity:` value for every model
   that ran. Otherwise the `Run GenAI accuracy` step fails. A skipped model (see
   below) does not fail the step.
2. On pull requests and merge groups, each model's perplexity may rise by at
   most 2% over its `<model>/result.json` in the same runner's results
   artifact of the latest successful `Windows Build` push run on `main` that
   has one. Lower perplexity is better. A model's baseline is used only when its
   model revision, framework commit, and test parameters match this run. Pushes
   report the delta without gating. A model with no comparable baseline is not
   gated.

Results appear in the job summary and, on PRs, in one comment per runner
marked `<!-- morphizen-<artifact name> -->`: one table with a row per model
(perplexity, main baseline, delta, samples, time, status), followed by notes for
skipped, failed, or non-comparable models. Each
results artifact has one subdirectory per model
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

The models live in the model store `--models-root` as `<models-root>/<model>`.
CI uses `<runner.workspace>/genai-accuracy/models`; the default is
`$GENAI_MODELS_ROOT`, or `C:\models` on Windows. `prepare` sets up the
framework, venv and wheels once, then for each model:

1. Uses the model in the store when `<models-root>/<model>/genai_config.json`
   exists and no `.download_incomplete` marker is left by an interrupted
   download. A model placed there by hand is used as is.
2. Otherwise, for a model with a Hugging Face source, downloads the pinned
   revision. The download is incremental, so an interrupted one resumes.
3. Otherwise uses the staged OGA/VLM benchmark copy under `--staged-root`
   (`<runner.workspace>` in CI) in place.
4. Links `<cache-root>/genai-models/<model>` to that directory: a symlink, or a
   directory junction when the account cannot create symlinks.

One model failing to download does not stop the others; `prepare` then exits
non-zero.

Each run hard-links the model into its work dir, because `run_accuracy.py`
overwrites `genai_config.json` in the model directory; the metadata files are
copied, so a staged model is never modified. The work dir must be on the same
volume as the model store and the staged models.

### Skipped models

A model with no Hugging Face source that is neither in the store nor staged is
skipped, not failed: `prepare` logs `[SKIP]` and emits a CI warning, `run` writes a
`result.json` with `skipped: true`, the report shows `SKIPPED`, and the job does
not gate on it.

## Running locally

On a Windows Strix Halo machine with Python 3.14 and an extracted
`gpu-test-package`:

```bash
python test/genai_accuracy/run_genai_accuracy.py prepare \
  --model Qwen3.8-Flash-Next Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --models-root D:/models --staged-root D:/runner-workspace
python test/genai_accuracy/run_genai_accuracy.py run \
  --model Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --models-root D:/models \
  --work-dir D:/genai-accuracy-work \
  --results-dir genai-accuracy-results/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml
python test/genai_accuracy/run_genai_accuracy.py report \
  --model Qwen3.8-Flash-Next Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml \
  --results-dir genai-accuracy-results --baseline-dir main \
  --summary-out summary.md --regressions-out regressions.txt
```

## Changing the pins

Changing `ACCURACY_REPO_COMMIT`, a model `revision`, a model's test parameters, or
the framework's `PPL`/`PPL_VLM` parameters makes the main baseline non-comparable
for the affected models. The first PR after such a change runs ungated for them.
Gating resumes once that change has a successful run on `main`.
