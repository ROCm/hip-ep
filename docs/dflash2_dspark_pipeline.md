<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# DSpark and DFlash2 speculative decoding for Qwen3.8-27B on hip-ep

How the DSpark and DFlash2 block drafters run as speculative decoders on hip-ep on the AMD
Strix Halo iGPU (gfx1151), driven by
[`scripts/run_dflash2_dspark_pipeline.py`](../scripts/run_dflash2_dspark_pipeline.py).

**Status.** On 240 Spec-Bench prompts (128 new tokens, baseline 13.36 tok/s), DFlash2 k=5 is
fastest at **2.01x**, ahead of DSpark k=5 at 1.70x and the model's MTP head (k=2) at 1.41x.
These are single passes, and the benchmark loop has a known EOS artifact (section 5), so
treat them as preliminary.

---

## Scope: what is in this repository

This change adds the pipeline script and this document. It makes **no hip-ep source change**:
the drafters are exported to ONNX graphs that hip-ep already runs under
`HIPDNN_EP_STRICT=1`, with no CPU fallback.

The speculative-decoding logic itself (the ONNX exporter, the draft/verify/accept loop and
the Spec-Bench driver) is host-side Python in an **external harness that is not part of
hip-ep**. The script orchestrates it and does not reimplement it. The script calls these
harness entry points, found in the directory given by `--harness` (or `$SPEC_HARNESS`):

| Stage | Harness entry point |
|---|---|
| `export` | `qwen36_fixups.py` (target tap graph), `dflash_export.py` (drafter ONNX) |
| `sanity`, `bench`, `summarize` | `qwen38_spec_bench.py` |
| `validate` (opt-in) | `dflash_capture.py`, `dflash_parity.py`, `dflash_runner.py` |

`preflight` checks that the `export` and `bench` entry points exist before anything runs.
The EAGLE-3 / MTP
pipeline ([`spec_decode_pipeline.md`](spec_decode_pipeline.md)) uses the same split.

---

## Quick start

You need:

- a hip-ep build with [PR #1043](https://github.com/ROCm/hip-ep/pull/1043), built per
  [`quick_start.md`](quick_start.md);
- the harness directory and a Python 3.13 environment with `onnx`, `onnxruntime`, `numpy`,
  `safetensors`, `tokenizers` and `huggingface_hub` (no torch);
- a models root holding the target with `mtp.onnx` beside it, and its embedding table
  (section 1);
- Strix Halo gfx1151 on Windows 11 and NTFS, and about 10 GB of free disk.

Run from the hip-ep checkout. `HARNESS`, `MODELS` and `BUILD` stand for your three paths:

```powershell
$common = "--preset", "qwen38-27b", "--harness", $HARNESS, "--models-root", $MODELS, `
          "--hipep-build", $BUILD, "--python", $PYTHON

# 1. check what is there (seconds, no GPU); every line should end OK, or TODO before export
python scripts/run_dflash2_dspark_pipeline.py @common --stages preflight

# 2. download the drafters at their pinned revisions and build the graphs (about 1 minute
#    after the download); preflight runs again afterwards
python scripts/run_dflash2_dspark_pipeline.py @common --fetch --stages preflight,export

# 3. smoke test: one prompt per subtask on every arm (about 2 minutes)
python scripts/run_dflash2_dspark_pipeline.py @common --stages preflight,sanity

# 4. the full Spec-Bench run (about 3 hours); the tables print at the end
python scripts/run_dflash2_dspark_pipeline.py @common --json results/qwen38_spec_bench.json

# 5. after an interruption, continue the same JSON (arguments after `--` go to the bench)
python scripts/run_dflash2_dspark_pipeline.py @common --json results/qwen38_spec_bench.json `
    --stages bench,summarize -- --resume
```

Preflight marks each check `OK`, `TODO` (export will build it), `MISS` (stop and fix; the
message says how) or `WARN`. Add `--dry-run` to print the commands without running them. When
the checkout sits next to the harness in a workspace laid out as `$WS/hip-ep` and
`$WS/tools/eagle3`, the four path flags can be omitted.

**Stages.** A failing stage stops the run. Stages always run in this order, whatever order
`--stages` lists them in.

| Stage | Does |
|---|---|
| `preflight` | Checks every input, the taps in the target graph, that each drafter was exported for the layers that graph taps, that `target.onnx.data` is a hard link to this target's weights, the EP library, the harness Python's packages and free disk. |
| `export` | Builds the tap graph and drafters if missing (`--fetch` downloads checkpoints). Re-exports a partial drafter; stops on a tap graph built with other flags. |
| `sanity` | One prompt per subtask, 16 tokens, every arm. |
| `bench` | The Spec-Bench run (`--json` required). Refuses an existing JSON unless `-- --resume` or `--overwrite`. Ctrl+C keeps the partial JSON and exits 130. |
| `summarize` | Re-prints the tables from the JSON, no GPU (skipped if `bench` just printed them). |
| `validate` | Opt-in, about 3 minutes: per drafter, ONNX-vs-reference parity and acceptance replay on hip-ep. |

**Presets.**

| Preset | Arms |
|---|---|
| `qwen38-27b` | `none, mtp:2, dspark:3, dspark:5, dflash2:3, dflash2:5` |
| `qwen38-27b-dflash2`, `qwen38-27b-dspark` | `none` plus that drafter at k=3 and 5 |
| `qwen36-27b`, `qwen36-27b-dflash2` | Qwen3.6-27B with the z-lab DFlash and community drafters; not covered here |

Other options: `--arms`, `--per-task`, `--max-new`, `--max-context`, `--no-strict`,
`--keep-env`, `--list-presets`. For a quick timed run, `--per-task 1 --max-new 32` takes about
2 minutes. The script removes every hip-ep tuning variable (`HIPDNN_EP_DEBUG`,
`HIPDNN_EP_PERF`, `HIPDNN_EP_MATMUL_*`, `HIPDNN_*_AUTOTUNE_MODE`) from its children and sets
`HIPDNN_EP_STRICT=1`, so a leaked variable cannot change what is measured. Run GPU
benchmarks serially.

---

## 1. Target and drafters

**Target.** AMD's `Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-dml-onnx` (HF tree
`3d00750a`), text graph only, in `<models-root>/models/amd_hipep_qwen38_27b/`. It takes
`inputs_embeds`, so the host looks embeddings up from an fp16 table at
`<models-root>/models/qwen3.8-27b-mtp-src/embed_tokens_fp16.npy`. `mtp.onnx` must already sit
beside the target; this script does not build it. Qwen3.8-27B is hybrid (16 attention and
48 Gated DeltaNet layers).

**Tap graph.** `export` writes `text_hipep_all.onnx` beside the target by structure-only
surgery that shares `text.onnx.data`. It unprunes `lm_head` so every verify row gets logits,
adds the MTP hidden taps, and adds one output `dflash_context [batch, seq, 25600]`: the
residuals entering layers 6, 20, 34, 48 and 62. One graph serves every arm, so the baseline
runs on identical weights.

**Drafters.**

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
target's embedding table and 248,320-token `lm_head`. Context rows are projected into a
persistent draft KV cache; the block attends non-causally over `[cache ; block]` and its rows
are discarded after each call. Both were trained against the FP8 target, so acceptance on
int4 is below the model cards.

**Export choices.** Each drafter directory under `<models-root>/models/qwen3.8-27b-spec/`
gets the graph (1.17 GB / 1.04 GB of weights), `draft_config.json`, `target.onnx.data` and,
for DFlash2, two 127 MB codebooks.

- **W4A16**: linear layers as int4 `MatMulNBits` (block 32), norms and gather tables fp16.
- **Shared `lm_head`**: `target.onnx.data` is an NTFS hard link to the target's 13.4 GB data,
  so nothing is copied. The export fails if the link cannot be made.
- **1/32 residual scale**: DFlash2's residual reaches about 1.5e5, past fp16's range. The
  scale is folded into the weights and RMSNorm eps, which is exact.
- **No `ArgMax` / `TopK`**: reduce-based argmax for DSpark and an exact two-stage top-16 for
  DFlash2 cut a call from 238 ms to 18 ms and from 79 ms to 17 ms.
- **Draft cache**: per layer, two chained `GroupQueryAttention` appends fill one buffer and a
  no-past `Attention` reads it under a host-built mask (section 3 explains why).

**Validation.** Against an fp32 reference using the same int4-rounded weights, the ONNX
drafters are within 1.3-1.5% relative L2 on the hidden states, with 21/21 (DSpark) and 19/21
(DFlash2, one near-tie) drafted tokens equal, on both CPU EP and hip-ep.

---

## 2. Runtime loop

`pending` holds tokens the target committed but has not consumed; its last element is the
**anchor**. Each iteration:

```
1. DRAFT   one draft call: append unseen context rows, draft the block from the anchor
2. VERIFY  one target call on pending + drafts
3. ACCEPT  keep the longest prefix that matches the target's argmax
4. COMMIT  on a rejection, rewind the target and carry the accepted tokens into the
           next verify call (DeltaNet state cannot be partially undone)
```

Every emitted token is the target's greedy argmax. Only accepted positions enter the draft
cache, so it never needs replaying. The target keeps all state on the device and
double-buffers the DeltaNet state so one call can be rewound. Verify width
(`pending + drafts`) is capped at 8 because width 8 costs 217 ms by default against 95 ms
with `HIPDNN_EP_MATMUL_DP4A_MAX_M=8`, which the runtime sets for its own process unless it is
already set. hip-ep compiles per input shape, so every verify width and draft context width
is warmed before timing.

---

## 3. hip-ep behaviours that shaped the export

All are worked around in the exporter.

| Behaviour | Workaround |
|---|---|
| `Attention` with a past, an additive mask and `is_causal=0` returns all zeros. | No-past `Attention` over a GQA-filled buffer. |
| An unconsumed, unexported node output crashes session creation (`morphizen-ep.cpp:408`). | Fold unused outputs into a `health` output. |
| `Concat` on a symbolic axis with a computed static input crashes at run time. | Two chained GQA appends. |
| `TopK` is slow (about 31 ms per row over the vocabulary); there is no `ArgMax`. | Reduce-based argmax, two-stage top-16. |
| `Attention` needs Q and K of the same rank and rejects a zero-length append. | Avoided by construction. |

See also [`supported-operations.md`](supported-operations.md).

---

## 4. Results

Qwen3.8-27B int4 gs128, hipgpu EP, IOBinding, strict, greedy, 128 new tokens, verify width
cap 8, `--preset qwen38-27b`.

**Spec-Bench, 240 prompts** (first 40 per subtask, single pass). "Clean" drops 10 prompts hit
by the EOS artifact (section 5); intervals are 95% bootstrap over prompts.

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

Speculative runs verify at width 8 and the baseline at width 1, so fp16 reductions can flip
a near-tie: 127-138 of 240 outputs per arm are token-identical to the baseline. Every
divergence checked was such a tie; not all were checked.

---

## 5. Known limitations

- **EOS artifact.** The benchmark loop stops only when the *last* pending token is EOS. If
  EOS lands mid-chain it runs on to `--max-new` and truncates, so a short answer shows about
  2 tok/s. It hit 10 of 240 prompts (9 translation) and is not fixed yet.
- Single passes only; no repeated runs, longer generations or longer contexts yet.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `preflight` reports a `harness:` line as MISS | `--harness` does not point at the harness directory. |
| `preflight` reports `harness python` MISS | That interpreter lacks a package the stage needs; install it or pass `--python`. |
| `has no output(s) ['dflash_context']` | The target graph was built without the DFlash context tap; delete it and rerun `export`. |
| Access violation or `morphizen-ep.cpp:408` at session creation | Unconsumed node output or symbolic `Concat` in a hand-edited graph (section 3). |
| `registered ... but reports no devices` | ROCm runtime not on `PATH`, or a session without GPU access. |
| Pages of JIT errors such as `Symbols not found: hipblasLtCreate` | An old `hipgpu.dll` was picked up without its ROCm runtime. Check the `hip-ep` line the script prints; pass `--hipep-build`. |
| Load error naming an unsupported node | CPU fallback is off by design; fix the node. |
| `draft cache overflow` / `KV overflow` | Raise `--max-context`. |
| Hard-link failure, or `preflight` reports the `lm_head link` is not a hard link | Delete the drafter's `target.onnx.data` and export on the same NTFS volume as the target. |
| `preflight` reports mismatched `tap layers` | The drafter and the tap graph disagree on the five layers. Use the preset that pairs them; it would otherwise run with low acceptance and no error. |
| `bench` refuses to start: the JSON already exists | Add `-- --resume` to continue it, or `--overwrite` to start over. |
| Verify ~217 ms at width 8 | `HIPDNN_EP_MATMUL_DP4A_MAX_M` below 8. |
| Inconsistent numbers | A leaked `HIPDNN_*` variable (`--keep-env` was passed) or another process on the GPU. |
| ~2 tok/s on a speculative arm, mostly translation | The EOS artifact (section 5). |

---

## 7. References

- This repository: [`spec_decode_pipeline.md`](spec_decode_pipeline.md),
  [`quick_start.md`](quick_start.md), [`supported-operations.md`](supported-operations.md),
  [PR #1043](https://github.com/ROCm/hip-ep/pull/1043).
- Drafter checkpoints: `RadixArk/Qwen3.8-27B-DSpark`, `incoai/Qwen3.8-27B-DFlash2` (Hugging
  Face, revisions in section 1).
- Reference implementations: SpecForge `dflash.py` / `dspark.py`; llama.cpp b11120
  (`src/models/dflash.cpp`, `common/speculative.cpp`).
