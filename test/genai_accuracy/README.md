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

| Item | Value |
|------|-------|
| Model | `Qwen3.8-Flash-Next` from `amd/Qwen3.8-Flash-Next_gpuep_1.0_gpu_128`, pinned revision |
| Test | `PPL_VLM`: conditional perplexity of `lmms-lab/flickr30k` test captions given the image; parameters come from the framework's `test_config_vlm.json` |
| EP | `AMDGPU` with `profile: hip` in every session, `session.disable_cpu_ep_fallback=1`, `search.chunk_size=512` |
| Framework | hipdnn-accuracy-test at a pinned commit |
| Python | 3.14 venv with the package's `onnxruntime_directml` and `onnxruntime_genai_directml` wheels |

`session.disable_cpu_ep_fallback=1` makes session creation fail if any node
would fall back to the CPU EP. A passing run therefore measured the GPU path.

## Pass criteria

1. `run_accuracy.py` exits 0 and reports a `Perplexity:` value. Otherwise the
   `Run GenAI accuracy` step fails.
2. On pull requests and merge groups, perplexity may rise by at most 2% over
   the `genai-accuracy-results` artifact of the latest successful `Windows
   Build` push run on `main`. Lower perplexity is better. The baseline is used
   only when its model revision, framework commit, and test parameters match
   this run. Pushes report the delta without gating.

Results appear in the job summary and, on PRs, in a comment marked
`<!-- morphizen-genai-accuracy-results -->`. The `genai-accuracy-results`
artifact contains `result.json`, the framework logs, and the generated
`genai_config_AMDGPU.json`.

## Runner layout

`--cache-root` (`<runner.workspace>/genai-accuracy` in CI) persists across runs:

```text
hipdnn-accuracy-test/   framework checkout at the pinned commit
genai-accuracy-venv/    venv, rebuilt when the Python or pinned packages change
hf-home/                Hugging Face cache (flickr30k)
genai-models/<repo>/    model download (~124 GB, downloaded once)
```

Each run hard-links the model into `--work-dir`, because `run_accuracy.py`
overwrites `genai_config.json` in the model directory. The work dir must be on
the same volume as the cache root.

## Running locally

On a Windows Strix Halo machine with Python 3.14 and an extracted
`gpu-test-package`:

```bash
python test/genai_accuracy/run_genai_accuracy.py prepare \
  --package-dir gpu-test-package --cache-root D:/genai-accuracy
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
