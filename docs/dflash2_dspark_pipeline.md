<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# DSpark and DFlash2 speculative decoding for Qwen3.8-27B on hip-ep

How the DSpark and DFlash2 block drafters are exported to ONNX and run as speculative
decoders on hip-ep on the AMD Strix Halo iGPU (gfx1151), driven by
[`scripts/run_dflash2_dspark_pipeline.py`](../scripts/run_dflash2_dspark_pipeline.py).

**Status.** On 240 Spec-Bench prompts (128 new tokens, baseline 13.36 tok/s), DFlash2 k=5 is
fastest at **2.01x**, ahead of DSpark k=5 at 1.70x and the model's MTP head (k=2) at 1.41x.
These are single passes, and the benchmark loops have a known EOS artifact (section 7), so
treat them as preliminary.

**Where the files live.** This repository holds only the EP, this document and the pipeline
script. Everything else is in a separate speculative-decoding workspace, `$WS`, with the
hip-ep checkout at `$WS/hip-ep`:

| Location | Contents |
|---|---|
| `$WS/tools/eagle3/` | the **harness**: exporters, runners, Spec-Bench driver (not hip-ep's own `tools/`) |
| `$WS/design/` | probe scripts, PowerShell wrappers, results, phase reports |
| `$WS/models/`, `$WS/data/` | target, drafters, embedding table (git-ignored); Spec-Bench questions |
| `$WS/build/hip-ep/` | a hip-ep build tree, written `$BUILD` |

The pipeline script runs from `$WS/hip-ep`; harness commands
(`.\.venv\Scripts\python.exe tools\eagle3\...`) run from `$WS`. The EAGLE-3 / MTP pipeline
([`spec_decode_pipeline.md`](spec_decode_pipeline.md)) uses the same split.

---

## Quick start

You need the workspace with its `.venv`, the target model with `mtp.onnx` and the embedding
table (section 2), and a hip-ep build in `$WS/build/hip-ep`. Run everything from `$WS/hip-ep`.
Any `python` will do: the script runs the harness with `$WS\.venv` and sets up `PATH` and the
EP for it.

```powershell
# 1. check what is there (seconds, no GPU); every line should end OK, or TODO before export
python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b --stages preflight

# 2. if anything is missing: download the drafters and build the graphs
#    (about 1 minute after the download), then preflight runs again
python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b --fetch --stages preflight,export

# 3. smoke test: one prompt per subtask on every arm (about 2 minutes)
python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b --stages preflight,sanity

# 4. the full Spec-Bench run (about 3 hours); the tables print at the end
python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b `
    --json ../design/results/qwen38/spec_bench.json

# 5. after an interruption, continue the same JSON (the arguments after `--` go to the bench)
python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b `
    --json ../design/results/qwen38/spec_bench.json --stages bench,summarize -- --resume
```

Preflight marks each check `OK`, `TODO` (export will build it), `MISS` (stop and fix; the
message says how) or `WARN`. To see the commands without running them, add `--dry-run`. If
the checkout is not at `$WS/hip-ep`, pass `--harness`, `--models-root` and `--hipep-build`.
Section 5 lists the stages and options, and section 8 covers errors.

---

## 1. Overview

DSpark and DFlash2 are **block drafters**: one draft call proposes up to 7 tokens, instead
of the k sequential calls of an MTP or EAGLE-3 head, conditioned on hidden states from five
intermediate target layers.

```
 HF BF16 checkpoint --> dflash_export.py --> W4A16 draft ONNX (+ host codebooks)
                                                        |
 AMD int4 target ONNX --> qwen36_fixups.py --> text_hipep_all.onnx (target + 5-layer tap
                                                                    + MTP taps)
                                                        v
                         dflash_runner.py / qwen38_spec_bench.py / qwen38_chat.py:
                         draft -> verify -> accept on hip-ep, IOBinding, device-resident caches
```

- All speculative logic is host-side Python in the harness; **no hip-ep source change**.
- Both drafters share one exporter, one runtime class (`BlockDrafter`) and one loop.
- CPU fallback is disabled, so an unsupported node fails loudly.
- The target's files are never modified; new graphs share its weights through a hard link.
- Every emitted token is the target's greedy argmax; outputs differ from k=0 only at fp16
  near-ties.

---

## 2. Environment and target

Requirements: Strix Halo gfx1151 on Windows 11 and NTFS; hip-ep with
[PR #1043](https://github.com/ROCm/hip-ep/pull/1043), built per
[`quick_start.md`](quick_start.md); `$WS/.venv` (Python 3.13) with `onnx`, `onnxruntime`,
`numpy`, `safetensors`, `tokenizers`, `huggingface_hub` (no torch); about 10 GB free disk.

```powershell
Set-Location $WS
$BUILD = "$WS\build\hip-ep"
$env:PATH = "$BUILD\_therock\bin;" + $env:PATH
$env:HIPEP_EP_BIN = "$BUILD\bin"
$env:PYTHONIOENCODING = "utf-8"
$env:HIPDNN_EP_STRICT = "1"      # fail if a graph cannot compile fully on the GPU
# before benchmarks, clear HIPDNN_EP_DEBUG, HIPDNN_EP_PERF and every HIPDNN_EP_MATMUL_* /
# HIPDNN_*_AUTOTUNE_MODE override; a leaked one changes what is measured
```

The pipeline script does all of this for its children; you only need it to run harness
commands by hand. Run GPU benchmarks serially.

**Target.** AMD's `Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-dml-onnx` (HF tree
`3d00750a`) in `$WS/models/amd_hipep_qwen38_27b/`, without the vision encoder. It takes
`inputs_embeds`, so the host looks embeddings up from
`$WS/models/qwen3.8-27b-mtp-src/embed_tokens_fp16.npy` (from the Qwen3.8 MTP work;
`qwen36_embedding.py` can rebuild it). `mtp.onnx` must already sit beside the target.

**Tap graph.** Structure-only surgery that shares `text.onnx.data`:

```powershell
.\.venv\Scripts\python.exe tools\eagle3\qwen36_fixups.py `
    --in-dir models\amd_hipep_qwen38_27b --in-model text.onnx --out text_hipep_all.onnx `
    --no-repair-mask-ref --no-mha-to-gqa `
    --unprune-lm-head --add-hidden-taps --add-dflash-context
```

`--unprune-lm-head` gives logits for every verify row; `--add-dflash-context` adds one output
`dflash_context [batch, seq, 25600]`, the residuals entering layers 6, 20, 34, 48 and 62
(`--dflash-layers` overrides); `--add-hidden-taps` feeds the MTP head, so one graph serves
every arm. Without it, and with `--out text_hipep_dflash.onnx`, you get the drafter-only graph
the single-prompt tools default to.

Qwen3.8-27B is hybrid (16 attention + 48 Gated DeltaNet layers). The target runtime keeps all
state on the device and double-buffers the DeltaNet state so one call can be rewound.

---

## 3. Drafters: checkpoints, export and validation

| | DSpark | DFlash2 |
|---|---|---|
| Repo | `RadixArk/Qwen3.8-27B-DSpark` | `incoai/Qwen3.8-27B-DFlash2` |
| Revision | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` | `015e795645c74b1a0eeef3b570031fb62e769bc5` |
| Params | 1.86 B | 1.92 B |
| Block | 7: `[anchor, MASK x6]` | 8: `[anchor, MASK x7]`, rows 1-7 drafted |
| RoPE, attention | YaRN, dense | default, sliding window 2048 |
| Token selection | Markov head (rank 256), greedy chain in the graph | top-16 per row + codebook selector walk on the host |
| Extra | - | 2-tap dynamic conv around attention and MLP |

Both are 5-layer Qwen3-style transformers (hidden 5120, 32 Q / 8 KV heads) that reuse the
target's embedding table and 248,320-token `lm_head`. Context rows `[*, 25600]` are projected
into a persistent draft KV cache; the block attends non-causally over `[cache ; block]` and
its rows are discarded after each call. Both were trained against the FP8 target, so
acceptance on int4 is below the model cards.

**Download and export.** `--fetch` downloads `model.safetensors` and `config.json` at the
pinned revisions into `$WS/models/qwen3.8-27b-spec/src/{dspark,dflash2}`. Then:

```powershell
.\.venv\Scripts\python.exe tools\eagle3\dflash_export.py `
    --src models\qwen3.8-27b-spec\src\dspark  --out-dir models\qwen3.8-27b-spec\dspark-w4a16
.\.venv\Scripts\python.exe tools\eagle3\dflash_export.py `
    --src models\qwen3.8-27b-spec\src\dflash2 --out-dir models\qwen3.8-27b-spec\dflash2-w4a16
```

Each output directory gets the graph (1.17 GB / 1.04 GB of weights), `draft_config.json`,
`target.onnx.data` and, for DFlash2, two 127 MB codebooks. Export choices:

- **W4A16**: linear layers as int4 `MatMulNBits` (block 32), norms and gather tables fp16.
- **Shared `lm_head`**: `target.onnx.data` is an NTFS hard link to the target's 13.4 GB data,
  so nothing is copied. The export fails if the link cannot be made.
- **1/32 residual scale**: DFlash2's residual reaches about 1.5e5, past fp16's range. The
  scale is folded into the weights and RMSNorm eps, which is exact.
- **No `ArgMax` / `TopK`**: reduce-based argmax for DSpark and an exact two-stage top-16 for
  DFlash2 cut a call from 238 ms to 18 ms and from 79 ms to 17 ms.
- **Draft cache**: per layer, two chained `GroupQueryAttention` appends fill one buffer and a
  no-past `Attention` reads it under a host-built mask (section 6 explains why).

**Validate** with `--ep hipgpu --iobinding` and `HIPDNN_EP_STRICT=1`, or the pipeline's
`validate` stage:

```powershell
# record a greedy trajectory with context features (--prompt-file for the report prompt)
.\.venv\Scripts\python.exe tools\eagle3\dflash_capture.py --ep hipgpu --iobinding --chat -n 256 `
    --target-dir models\amd_hipep_qwen38_27b --target text_hipep_dflash.onnx `
    --embed models\qwen3.8-27b-mtp-src\embed_tokens_fp16.npy `
    --tokenizer-dir models\amd_hipep_qwen38_27b --out models\qwen3.8-27b-spec\ref\traj_phasem.npz
# ONNX drafter vs the fp32 numpy reference (dflash_ref.py), same int4 weights
.\.venv\Scripts\python.exe tools\eagle3\dflash_parity.py --ep hipgpu --iobinding --ref-quant `
    --draft-dir models\qwen3.8-27b-spec\dspark-w4a16 --src models\qwen3.8-27b-spec\src\dspark `
    --traj models\qwen3.8-27b-spec\ref\traj_report.npz
# acceptance by depth on hip-ep
.\.venv\Scripts\python.exe tools\eagle3\dflash_runner.py --ep hipgpu --iobinding `
    --draft-dir models\qwen3.8-27b-spec\dflash2-w4a16 --replay models\qwen3.8-27b-spec\ref\traj_phasem.npz
```

Parity against the int4-rounded reference is 1.3-1.5% relative L2 on the hidden states, with
21/21 (DSpark) and 19/21 (DFlash2, one near-tie) drafted tokens equal, on both CPU EP and
hip-ep. Speculative runs verify at width 8 and k=0 at width 1, so fp16 reductions can flip a
near-tie; `qwen38_verify_margin.py` checks a divergence is a tie, and every one checked was.

---

## 4. Runtime

`pending` holds tokens the target committed but has not consumed; its last element is the
**anchor**. Each iteration:

```
1. DRAFT   one draft call: append unseen context rows, draft the block from the anchor
2. VERIFY  one target call on pending + drafts
3. ACCEPT  keep the longest prefix that matches the target's argmax
4. COMMIT  on a rejection, rewind the target and carry the accepted tokens into the
           next verify call (DeltaNet state cannot be partially undone)
```

Only accepted positions enter the draft cache, so it never needs replaying. Verify width
(`pending + drafts`) is capped at 8 because width 8 costs 217 ms by default against 95 ms with
`HIPDNN_EP_MATMUL_DP4A_MAX_M=8`; the runner sets that variable for its own process unless it
is already set. hip-ep compiles per input shape, so the runner warms every verify width and
draft context width before timing.

---

## 5. Running it

**Single prompt**, sweeping k (0 is the baseline):

```powershell
.\.venv\Scripts\python.exe tools\eagle3\dflash_runner.py --ep hipgpu --iobinding --chat `
    --draft-dir models\qwen3.8-27b-spec\dflash2-w4a16 `
    --sweep 0,3,5,7 -n 128 --warmup 16 --max-context 1024 `
    --json design\results\qwen38\dflash_e2e_dflash2_phasem.json
```

Useful options: `--prompt` / `--prompt-file`, `--width-cap` (8), `--max-pending` (8),
`--dp4a-max-m` (8, 0 leaves the default), `--verify-text` (abort on divergence), `--show`.

**Spec-Bench**, all arms paired per prompt on one loaded target:

```powershell
.\.venv\Scripts\python.exe tools\eagle3\qwen38_spec_bench.py --ep hipgpu --iobinding `
    --arms none,mtp:2,dspark:3,dspark:5,dflash2:3,dflash2:5 --per-task 40 -n 128 `
    --max-context 1792 --json design\results\qwen38\spec_bench.json
```

Arms are `none | mtp:<k> | dspark:<k> | dflash2:<k>`, optionally `:cap<N>` or `:mp<N>`.
`--per-task N` takes the first N prompts of each of the six subtasks. The JSON is rewritten
per prompt, so `--resume` continues and `--summarize-only` re-prints. 240 prompts take about
3 hours.

**Chat**: `qwen38_chat.py --ep hipgpu --iobinding --head dflash2`, with `/head`, `/k`,
`/compare`, `/think` and `/stats`.

**Pipeline script.** Each stage runs a harness command; a failing stage stops the run.
Stages always run in the order below, whatever order `--stages` lists them in.

| Stage | Does |
|---|---|
| `preflight` | Checks every input, the taps in the target graph, that each drafter was exported for the layers that graph taps, that `target.onnx.data` is a hard link to this target's weights, the build tree and free disk. |
| `export` | Builds the tap graph and drafters if missing (`--fetch` downloads checkpoints). Re-exports a partial drafter; stops on a tap graph built with other flags. |
| `sanity` | One prompt per subtask, 16 tokens, every arm. |
| `bench` | The Spec-Bench run (`--json` required). Refuses an existing JSON unless `-- --resume` or `--overwrite`. |
| `summarize` | Re-prints the tables from the JSON, no GPU (skipped if `bench` just printed them). |
| `validate` | Opt-in, about 3 minutes: trajectories, parity and replay per drafter. |

| Preset | Arms |
|---|---|
| `qwen38-27b` | `none, mtp:2, dspark:3, dspark:5, dflash2:3, dflash2:5` |
| `qwen38-27b-dflash2`, `qwen38-27b-dspark` | `none` plus that drafter at k=3 and 5 |
| `qwen36-27b`, `qwen36-27b-dflash2` | Qwen3.6-27B with z-lab DFlash and community drafters; not covered here |

Options: `--harness`, `--models-root`, `--hipep-build`, `--python`, `--stages`, `--json`,
`--overwrite`, `--arms` / `--per-task` / `--max-new` / `--max-context` overrides, `--fetch`,
`--no-strict`, `--keep-env`, `--dry-run`, `--list-presets`. `--python` defaults to
`$WS\.venv` and `--hipep-build` to `$WS\build\hip-ep` (or `$HIP_EP_BUILD`). Arguments after
`--` go to `qwen38_spec_bench.py` (for example `-- --resume` or `-- --only-task qa`). Ctrl+C
keeps what the bench has written and exits 130. Every stage has been run end to end on a
fresh models root. For a quick timed run, `--per-task 1 --max-new 32` takes about 2 minutes.

---

## 6. hip-ep behaviours that shaped the design

Found with `$WS/design/scripts/dflash_probe_attn.py`; all are worked around in the exporter.

| Behaviour | Workaround |
|---|---|
| `Attention` with a past, an additive mask and `is_causal=0` returns all zeros. | No-past `Attention` over a GQA-filled buffer. |
| An unconsumed, unexported node output crashes session creation (`morphizen-ep.cpp:408`). | Fold unused outputs into a `health` output. |
| `Concat` on a symbolic axis with a computed static input crashes at run time. | Two chained GQA appends. |
| `TopK` is slow (about 31 ms per row over the vocabulary); there is no `ArgMax`. | Reduce-based argmax, two-stage top-16. |
| `Attention` needs Q and K of the same rank and rejects a zero-length append. | Avoided by construction. |

See also [`supported-operations.md`](supported-operations.md).

---

## 7. Results

Qwen3.8-27B int4 gs128, hipgpu EP, strict, greedy, 128 new tokens, verify width cap 8. Results
are in `$WS/design/results/qwen38/`.

**Spec-Bench, 240 prompts** (40 per subtask, single pass). "Clean" drops 10 prompts hit by the
EOS artifact below; intervals are 95% bootstrap over prompts.

| Arm | tok/s | speedup (CI) | clean speedup (CI) |
|---|---|---|---|
| Baseline | 13.36 | 1.00x | 1.00x |
| MTP k=2 | 18.79 | 1.41x (1.37-1.44) | 1.45x (1.43-1.47) |
| DSpark k=3 | 21.17 | 1.59x (1.54-1.63) | 1.59x (1.56-1.63) |
| DSpark k=5 | 22.77 | 1.70x (1.65-1.76) | 1.73x (1.68-1.79) |
| DFlash2 k=3 | 23.54 | 1.76x (1.72-1.81) | 1.78x (1.73-1.82) |
| **DFlash2 k=5** | **26.85** | **2.01x (1.94-2.08)** | **2.04x (1.97-2.11)** |

| Subtask | MTP k=2 | DSpark k=5 | DFlash2 k=5 |
|---|---|---|---|
| math_reasoning | 1.65x | 2.22x | **2.85x** |
| rag | 1.46x | 1.94x | 2.23x |
| summarization | 1.37x | 1.62x | 1.82x |
| qa | 1.35x | 1.45x | 1.68x |
| mt_bench | 1.45x | 1.63x | 1.97x |
| translation (clean, 31) | 1.40x | 1.65x | 1.84x |

DFlash2 k=5 wins every subtask, and k=5 beats k=3 for both drafters (k=6 and 7 were not run).
Acceptance by depth for DFlash2 k=5 is 83 / 63 / 46 / 33 / 25%.

**Single prompts.** On the Phase-M prompt, DFlash2 k=3 reaches 23.68 tok/s (1.79x) and DSpark
k=3 18.42 tok/s (1.39x), against 1.52x for MTP k=2. Offline acceptance tracks the llama.cpp
report closely at k <= 4.

**EOS artifact.** The benchmark loops stop only when the *last* pending token is EOS. If EOS
lands mid-chain they run on to `--max-new` and truncate, so a short answer shows about
2 tok/s. It hit 10 of 240 prompts (9 translation) and is not fixed yet; the chat tool already
stops at the first EOS.

Not yet done: repeated runs, longer generations and contexts, tie-checking every Spec-Bench
divergence (127-138 of 240 outputs per arm are token-identical to the baseline).

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `has no output(s) ['dflash_context']` | Target graph built without `--add-dflash-context` (section 2). |
| Access violation or `morphizen-ep.cpp:408` at session creation | Unconsumed node output or symbolic `Concat` in a hand-edited graph (section 6). |
| `registered ... but reports no devices` | ROCm runtime not on `PATH`, or a session without GPU access. |
| Pages of JIT errors such as `Symbols not found: hipblasLtCreate` | An old `hipgpu.dll` was picked up (for example from `$WS\local\bin`) without its ROCm runtime. Check the `hip-ep` line the script prints; pass `--hipep-build`. |
| `preflight` reports `harness python` MISS | That interpreter lacks a package the stage needs; install it in `$WS\.venv` or pass `--python`. |
| Load error naming an unsupported node | CPU fallback is off by design; fix the node. |
| `draft cache overflow` / `KV overflow` | Raise `--max-context`. |
| Hard-link failure, or `preflight` reports the `lm_head link` is not a hard link | Delete the drafter's `target.onnx.data` and export on the same NTFS volume as the target. |
| `preflight` reports mismatched `tap layers` | The drafter and the tap graph disagree on the five layers (for example a Qwen3.8-tap drafter on a Qwen3.6-tap graph). Use the preset that pairs them; it would otherwise run with low acceptance and no error. |
| `bench` refuses to start: the JSON already exists | Add `-- --resume` to continue it, or `--overwrite` to start over. |
| NaN / inf, large `draft_health` | Graph predates the residual scale; re-export. |
| DSpark call ~240 ms, DFlash2 ~79 ms or more | Exported with `--argmax topk`; re-export with the default. |
| Verify ~217 ms at width 8 | `HIPDNN_EP_MATMUL_DP4A_MAX_M` below 8. |
| Inconsistent numbers | A leaked `HIPDNN_*` variable or another process on the GPU. |
| ~2 tok/s on a speculative arm, mostly translation | The EOS artifact (section 7). |

---

## 9. References

- This repository: [`spec_decode_pipeline.md`](spec_decode_pipeline.md),
  [`quick_start.md`](quick_start.md), [`supported-operations.md`](supported-operations.md),
  [PR #1043](https://github.com/ROCm/hip-ep/pull/1043).
- Workspace: project notes in `$WS/.cursor/skills/hip-ep-dspark-dflash2/`, interim report
  `$WS/design/PHASE_W_QWEN38_DSPARK_DFLASH2.md`, earlier phases `$WS/design/PHASE_*.md`.
- External: SpecForge `dflash.py` / `dspark.py`; llama.cpp b11120 (`src/models/dflash.cpp`,
  `common/speculative.cpp`).
