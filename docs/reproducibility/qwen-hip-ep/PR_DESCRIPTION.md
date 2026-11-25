<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add frozen native QSA specializations and reproducible experiment evidence

<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
## Latest eight-selector performance

This performance addendum belongs to local branch `codex/qwen-eight-qsa-perf`; the historical four-selector merge-candidate validation below remains separately identified.

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Eight vs four: prefill | Eight vs four: decode |
|---|---:|---:|
| 4K | -5.21% | +0.71% |
| 16K | -6.31% | +1.94% |
| 32K | +0.92% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers.

The fixed 50-case BFCL component pilot scores Eight 29/50 versus reused Original 26/50; no new Original generation was run. This is not the official overall BFCL score. Eight GSM8K and application-memory comparisons remain unmeasured. The source checkpoint still misses its fixed 32K PPL gate; this pilot does not establish accepted long-context quality. See the [eight-selector report](../../experiments/qwen-qsa-eight-selector-results.md) and [raw-data paper handoff](../qwen-eight-hip-ep/README.md).
<!-- QWEN_EIGHT_PERFORMANCE_END -->

Replace four selected full-attention nodes with native stateful QSA operations
for the pinned OLMo and Qwen models. Selector projection, features, stable ranking
and selected-address attention run on the EP stream inside the existing full-model
session. Dense GQA remains available and provides Qwen's full-support path. Add
BF16 selector import, compiler/runtime validation and matched evaluation tools.

The Qwen W4A16 comparison measures batch1 at4K/16K/32K: decode throughput changes
by+0.50%/+1.23%/+1.60%, while prefill decreases4.13%/4.47%/8.83%. PPL changes
5.96868807 to5.97127044; GSM8K is111/128 for both, with identical short-context
generations. Full KV storage is retained and QSA adds workspace. These measurements
belong to the preserved pre-merge binaries, not the new merge build.

The paired50-case BFCLv4 component pilot scores default26/50 versus QSA24/50. The report records actual prompt/response lengths, turns, generation steps, cap counts and sparse-pruning coverage; it is not the official BFCLv4 overall score.

Include a portable JSON/CSV paper packet, compressed raw quality rows, raw paired
timings, checksum manifest and local raw archive descriptor. Explicitly record
memory coverage, the failed/canceled64K attempt, limited accuracy coverage and
precision/generation/statistics differences from the MI355X source campaign.

Validation: 736 compiler lit tests pass (nine unsupported), eight CPU unit tests
pass, six CPU export tests pass, and the expanded kernel/operator/full-model
controls pass on the merged candidate. Full-support equality and replay are
exact; both variants retain one EP dispatch per forward and stable cache
pointers. See [AUDIT.md](AUDIT.md),[candidate-validation.json](candidate-validation.json)
and the hashed historical receipts. The implementation is qualified for gfx1151/wave32; shape
guards do not establish full-model performance or memory support at all bounds.
The local PR branch resolves conflicts against local`origin/main` while preserving
upstream GQA host-length/bias handling, FP8 import and ROCTx tracing. No branch,
pull request or results have been pushed or published by this work.
