<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Manual multi-turn TurboQuant characterization

This Qwen3.8-27B test is deliberately absent from CTest, the default build tests
and pytest discovery. The Python entry point runs only when explicitly invoked.
Normal builds compile Ring and TurboQuant kernels; neither changes normal runtime
dispatch without opt-in. CPU-only metric/fixture unit tests remain cheap.

Use one HIP-EP build for the three OGA paths, with an optional independent BF16 reference:

| Path | Ring environment | Cache / prefill / decode |
|---|---|---|
| `pytorch-bf16` | not applicable | BF16 weights/cache, PyTorch SDPA, persistent hybrid cache |
| `no-ring` | `HIPDNN_EP_USE_RING_ATTENTION=0` | FP16 / upstream GQA / upstream GQA |
| `ring` | `HIPDNN_EP_USE_RING_ATTENTION=1` | FP16 / Ring / Ring |
| `turboquant` | `HIPDNN_EP_USE_RING_ATTENTION=1` plus codec provider options | packed 4-bit / Ring with mixed-cache append / TurboQuant |

Both `HIP_EP_ENABLE_RING_ATTENTION` and `HIP_EP_ENABLE_TURBOQUANT` default ON.
Use a compatible GenAI with AMDGPU continuous append and pending-token handling;
see [the companion patch](../../tools/turboquant/patches/onnxruntime-genai-0.17-amdgpu-append.patch).
Generate codec assets during build/setup, never add BIN/NPZ/model/logit files to Git.
The harness requires the Qwen geometry (24Q/4KV/D256 and 16 full-attention layers).

From the repository root, in the prepared MSVC/Python/ROCm environment:

```powershell
python test/long_context/run_turboquant_multiturn.py `
  --model C:/work/models/qwen-original `
  --install C:/work/install/turboquant `
  --codec C:/work/reference/codec/codec.bin `
  --dll-dir C:/path/to/rocm/core/bin --dll-dir C:/path/to/rocm/devel/bin `
  --turns 8 --prompt-tokens 4096 --max-new-tokens 128 `
  --modes no-ring,ring,turboquant --tracks matched-history,free-running `
  --torch-model C:/work/models/Qwen3.8-27B-bf16 `
  --torch-python C:/work/torch-reference-venv/Scripts/python.exe `
  --output C:/work/results/multiturn-8x4k
```

Use a new output directory on the model's volume (the harness hard-links weights).
The supplied interpreter/PYTHONPATH determines GenAI. Provider libraries come
from `--install`; record both versions. Workers run serially in fresh processes
with private model/cache directories. Run no other GPU jobs or builds concurrently.
Strict offload and dispatch-log assertions are mandatory. Diagnostic timings
include tracing, synchronization/readback and do not replace performance benchmarks.

Each turn appends exactly 4,096 **new prompt tokens including chat delimiters**.
The fictional project ledger includes delayed recall, revisions, arithmetic,
contradictory rumors, unknown information and format requirements. Archive filler
does not change ledger facts. This is a small synthetic characterization, not
MT-Bench, LongMemEval or a broad model-quality evaluation.

The fixture uses the supplied chat template's non-thinking assistant prefix.
Every path uses explicit argmax selection from raw logits with the same stop
tokens. If the answer reaches the token budget, one forced chat-end token seals
the turn; the receipt identifies it, and numerical comparisons include it.

Eight turns contribute 32,768 prompt tokens, plus up to 1,024 generated tokens
and up to eight forced chat-end tokens;
capacity includes those tokens and 256 spare slots. An EOS can end a continuation
early. One Generator persists across the eight appends. Before each append, the
last selected token is materialized, then all committed full-attention cache rows
are hashed and checked unchanged afterward. Pending-token lifecycle behavior is
also covered separately by the small GenAI regression.

The matched-history track first records the no-Ring greedy token choices, then
teacher-forces exactly those choices through Ring, TurboQuant, optional BF16
PyTorch, and a second no-Ring run. Hashes enforce identical histories. It compares the last prefill logits and
every subsequent next-token distribution: cosine, relative L2, maximum error,
KL/JS divergence, top-1 agreement, top-5 overlap, dense top-token margin, and
reference-token NLL change. The repeat measures dense numerical variation. No
quality threshold is fitted after observing results. Hard failures are non-finite
values, incorrect dispatch/positions, mutated old cache rows or mismatched histories.

The free-running track preserves each path's own answers. Numerical comparisons
across divergent histories are intentionally not made. Required-content checks
are diagnostic only; they do not catch every contradictory or unsupported claim.

```powershell
python test/long_context/judge_multiturn.py --results C:/work/results/multiturn-8x4k
```

This exports blinded three- or four-candidate packets and a separate key. Labels
rotate across turns; each turn is presented again in reverse candidate order.
An optional `--judge-model <pinned-version> --command <adapter> <args...>` runs a
judge adapter via JSON stdin/stdout. Use an independent model with enough context
for the full shared user history (over 32K at turn eight), every candidate's
prior answers and the rubric. Use reproducible settings and no
silent truncation. Record the adapter, model version and prompt hashes; report
order disagreement and per-turn scores alongside deterministic checks. The
requested judge is `gpt-6.1-sol`; disclose its actual invocation/settings instead
of claiming an unobserved API version or temperature. LLM scores
supplement numerical/kernel checks. Inspect outputs before treating them as evidence.

`manifest.json` records source diff, libraries, model/tokenizer, codec and fixture
hashes. `comparisons.json` has per-position metrics; `summary.md` has the per-turn
table. Preserve raw artifacts outside Git and add measured findings, limitations
and reproduction commands to the PR document. Do not report an unrun judge as passed.

### Current upstream GenAI stack

Current upstream CI applies GenAI PR #2691, which already adds AMDGPU continuous
decoding. Do **not** apply the full stock-0.17 patch again on that stack. Apply
only `tools/turboquant/patches/onnxruntime-genai-0.17-pending-append.patch` there
(or use `git apply --include='src/decoding/*'` on the full patch). The current
2685/2691/2705/2712/2792/2801 patch stack does not modify either standard-decoding
file; the remaining patch is apply-checked against v0.17.0. It changes only
pending-token flush before append, independent of KV precision. No rewind change
is included. Local results identify the exact GenAI package used; applying a
patch is not a claim that every CI dependency combination has been tested.

### Optional BF16 reference

Omit both `--torch-model` and `--torch-python` for the three OGA paths alone.
For four paths, use the matching original BF16 checkpoint and an isolated Python
environment with GPU-enabled PyTorch and Transformers 5.22.1. Earlier versions
may reset Qwen3.5 recurrent state on a multi-token append; the reference checks
the state-aware implementation and records its source hash. It validates exact
vocabulary-to-ID equality, BF16 GPU parameter placement, cache length, finite
logits, unchanged committed BF16 KV rows, and a short append-versus-full-replay
sanity comparison. No CPU offload or custom model/kernel patch is used.

The BF16 reference loads the original conditional-generation checkpoint; vision
parameters are resident but unused for text inference. Its dynamic cache reports
live KV bytes, while OGA reports reserved cache capacity. Do not compare these as
the same memory metric. Diagnostic timings also reflect different framework and
readback overheads. All reference weight/config and implementation hashes are
recorded in its receipt.

The three OGA arms share the same INT4 weights. Ring versus TurboQuant is the
closest isolation of KV compression effects; BF16 versus any OGA arm additionally
includes weight quantization and framework/kernel arithmetic differences. Report
these comparisons separately. This eight-question ledger fixture cannot establish
general long-context answer quality, even when the judge assigns perfect scores.
