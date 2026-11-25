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

The workflow runs these models in order, each through the same steps:

| Model | Source | Revision |
|-------|--------|----------|
| `Qwen3.8-Flash-Next` | Hugging Face `amd/Qwen3.8-Flash-Next_gpuep_1.0_gpu_128` | pinned |
| `Qwen3.8-27B` | Hugging Face `amd/Qwen3.8-27B_gpuep_1.0_gpu_128` | pinned |
| `Gemma-4-E4B-it` | modelzoo entry `gemma-4-E4B-it-awq-uint4-onnx-split-fp16-pruned_lm_head` (not on Hugging Face), staged by hand | n/a |

All models use the `PPL_VLM` test, matching the modelzoo `vlm_model_mm_accuracy`
profile. Its parameters come from the framework's `test_config_vlm.json`:
conditional perplexity of `lmms-lab/flickr30k` test captions given the image.

| Item | Value |
|------|-------|
| EP | `AMDGPU` with `profile: hip` in every session, `session.disable_cpu_ep_fallback=1`, `search.chunk_size=512` (`Gemma-4-E4B-it` sets no chunk size and uses `enable_graph_capture=0`) |
| Framework | hipdnn-accuracy-test at a pinned commit |
| Python | 3.14 venv with the package's `onnxruntime_directml` and `onnxruntime_genai_directml` wheels |

The LLM models in the modelzoo (`gpt-oss-20b`, `Llama-3.1-8B`) are not
included.

`session.disable_cpu_ep_fallback=1` makes session creation fail if any node
would fall back to the CPU EP. A passing run therefore measured the GPU path.

## Pass criteria

1. `run_accuracy.py` exits 0 and reports a `Perplexity:` value for every model.
   Otherwise the `Run GenAI accuracy` step fails.
2. On pull requests and merge groups, each model's perplexity may rise by at
   most 2% over its `<model>/result.json` in the `genai-accuracy-results`
   artifact of the latest successful `Windows Build` push run on `main` that
   has one. Lower perplexity is better. A model's baseline is used only when its
   model revision, framework commit, and test parameters match this run. Pushes
   report the delta without gating. The first run after the artifact layout
   changed has no baseline, so it is not gated.

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
hf-home/                Hugging Face cache (flickr30k)
genai-models/<model>    link to the model in the model store
```

The models live in the machine-wide model store `--models-root` (default
`$GENAI_MODELS_ROOT`, or `C:\models` on Windows) as `<models-root>/<model>`, so
every runner workspace shares one copy. The three models take about 150 GB
together (Flash-Next ~124 GB, Qwen3.8-27B ~17 GB, Gemma ~11 GB). `prepare`:

1. Treats the model as present when `<models-root>/<model>/genai_config.json`
   exists and no `.download_incomplete` marker is left by an interrupted
   download. A model placed there by hand is used as is; its revision is not
   verified against the pin.
2. Otherwise downloads the pinned revision into `<models-root>/<model>`. A model
   without a Hugging Face source (`Gemma-4-E4B-it`) is not downloaded; when it is
   absent from the store its test is skipped (see below).
3. Links `<cache-root>/genai-models/<model>` to it: a symlink, or a directory
   junction when the account cannot create symlinks.

Each run hard-links the model into its work dir, because `run_accuracy.py`
overwrites `genai_config.json` in the model directory. The work dir must be on
the same volume as the model store.

### Staging Gemma-4-E4B-it

Copy the modelzoo directory from `xcosdaem579.xilinx.com` (the `model_url` of the
modelzoo entry, `/proj/aiebuilds/vitisaijenkins/ROCm/models/key_llm/gemma-4-E4B-it-awq-uint4-onnx-split-fp16-pruned_lm_head`,
about 11 GB) into `C:\models\Gemma-4-E4B-it` on the runner. The directory must
contain `genai_config.json`. Until it is there, this model's test is skipped: `run`
writes a `result.json` with `skipped: true`, the report shows `SKIPPED`, and the
job does not fail or gate on it.

## Running locally

On a Windows Strix Halo machine with Python 3.14 and an extracted
`gpu-test-package`:

```bash
python test/genai_accuracy/run_genai_accuracy.py prepare \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --models-root D:/models
python test/genai_accuracy/run_genai_accuracy.py run \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy \
  --work-dir D:/genai-accuracy-work --results-dir genai-accuracy-results
python test/genai_accuracy/run_genai_accuracy.py report \
  --current genai-accuracy-results/result.json \
  --baseline main/result.json \
  --summary-out summary.md --regressions-out regressions.txt
```

## Changing the pins

Changing `ACCURACY_REPO_COMMIT`, a model `revision`, or the framework's
PPL_VLM parameters makes the main baseline non-comparable. The first PR after
such a change runs ungated. Gating resumes once that change has a successful
run on `main`.
