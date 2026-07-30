<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Prune Logits

Offline transform that prunes an ONNX decoder model so its `lm_head` runs
on the **last token only**, collapsing the `logits` output from
`[batch, sequence_length, vocab]` to `[batch, 1, vocab]`.

During autoregressive generation only the last position's logits are consumed, so
full-sequence logits waste a large logits buffer (+ `lm_head` dequant scratch) and
a sequence-wide matmul at long prefill. Pruning them cuts prefill time-to-first-
token and peak memory, with no effect on decode (already one token).

## What it does

```
Before:  final_norm -[B, seq, H]-> lm_head(MatMul*) -> logits [B, seq, vocab]
After:   final_norm -[B, seq, H]-> Gather(idx=-1, axis=1) -[B, H]->
             Unsqueeze(axes=[1]) -[B, 1, H]-> lm_head -> logits [B, 1, vocab]
```

The rewrite is purely structural. External weights (`*.onnx_data*`) are **never**
rewritten — the tool loads structure only and shares the original data files.

## Install

```bash
pip install -r requirements.txt
```

`onnxruntime` is **not** in the core requirements; install it separately only if
you want the optional `--ort-verify` numeric check:

```bash
pip install onnxruntime
```

## Usage

The argument is always a model **directory** (not a bare `.onnx`):

```bash
# Apply (one-shot: pre-check -> backup -> transform -> structural review)
python prune_logits.py <model_dir>

# Eligibility check + plan only, no writes
python prune_logits.py <model_dir> --precheck

# Also verify last-token logits equivalence with onnxruntime (opt-in)
python prune_logits.py <model_dir> --ort-verify

# Revert to the original model from the *_orig backup
python prune_logits.py <model_dir> --restore
```

The directory resolves the target `.onnx` from `genai_config.json`
(`model.decoder.filename`), so on multi-part exports (e.g. VLMs with separate
vision / embedding / decoder graphs) **only the text/decoder graph is ever
selected** — vision and embedding graphs are never read or modified. When no
`genai_config.json` is present, the tool falls back to the single
non-`*_orig.onnx` in the folder, and errors out if that is ambiguous. A bare
`.onnx` path is intentionally **not** accepted, to keep decoder resolution
unambiguous and prevent accidentally targeting a non-decoder graph.

## Safety model

- **Backup == rename.** The original `<model>.onnx` is renamed to
  `<model>_orig.onnx`; the pruned graph is written under the **original**
  filename. Because the model keeps its name, `genai_config.json` needs no change
  and is never touched.
- **Fail closed.** A pre-check failure leaves the model untouched. Any failure
  *after* writing (structural review or `--ort-verify`) auto-rolls-back to the
  original — nothing partial is kept.
- **Idempotent.** Re-running when `_orig` already exists (or the model is already
  pruned) is refused, leaving the model as-is; use `--restore` to revert.
- **Fail fast on ORT.** `--ort-verify` requires `onnxruntime`; its absence is
  detected in pre-flight and aborts before any change.

## Options

| Option | Purpose |
|---|---|
| `--logits-name NAME` | Graph logits output name (default `logits`) |
| `--lm-head-name NAME` | Override lm_head node auto-detection |
| `--ort-verify` | Numeric last-token equivalence check via onnxruntime (off by default) |
| `--precheck` | Eligibility check + plan only; no writes |
| `--restore` | Revert from the `*_orig` backup |

## Eligibility

A model is eligible when:

- it has a rank-3 `logits` output `[batch, sequence, vocab]` whose sequence dim is
  **not** already `1`;
- a single node (a `MatMul` / `MatMulNBits` / `Gemm` family op, or a node whose
  name contains `lm_head`) produces `logits`, and its first input is the hidden
  state.

Use `--lm-head-name` / `--logits-name` for non-standard graphs. If the producer
op is not a typical lm_head op the tool warns but still proceeds (the transform
only needs the node's first input to be the hidden state).

## `--ort-verify` details

Runs in an **isolated subprocess** (so an out-of-memory kill cannot corrupt the
apply flow). It builds a tiny dummy prompt (8 tokens, empty past-KV),
runs the `_orig` and pruned models on `CPUExecutionProvider`, and compares the
original's last-token logits (`logits[:, -1, :]`) against the pruned output
(`[:, 0, :]`): `argmax`, top-5 overlap, max-abs-diff, and cosine. The last-token
slice is mathematically identical, so it should match within fp rounding.

> Note: this loads full model weights on CPU (twice, sequentially). For very large
> models it is memory-heavy — hence it is opt-in.
