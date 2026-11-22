<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Frozen Qwen QSA at the native GQA boundary

This is the implementation contract for the integrated native port. Matched
default/QSA quality and batch-one performance measurements are complete at the
qualified contexts. Current support, numerical controls, source identities and
measurement limits are in the
[evidence record](../experiments/qwen-qsa-native-gqa-results.md) and the
[audit and paper handoff](../reproducibility/qwen-hip-ep/README.md).

## Scope and execution boundary

Replace only full-attention layers **27/31/35/39**, zero-based, in the pinned
Qwen/Qwen3.8-27B revision. Preserve all 48 Gated DeltaNet blocks, other full
attention, QKV/O projections, native Q/K norms, native partial RoPE, output gate,
embedding, output head and full KV storage. Both variants use the same fresh
RTN W4A16/FP16 backbone bytes and FP32 recurrent state.

Use a distinct native GQA specialization with one full-model ONNX session and
one EP dispatch per forward. No Python attention replacement, partitioned model
sessions, boundary tensor round trips or full-history KV gather/copy. Internal
feature, ranking and selected-address attention launches are part of this native
operation. The previous OLMo specialization and its qualified geometry remain
unchanged.

The extra `output_gate_type=swish` metadata does not describe the executed
Transformers 5.19 Qwen3_5Attention gate. The native default and frozen source
experiment execute **sigmoid**; the CPU control verifies this. Preserve that
executed behavior outside the replacement region.

## Tensor and mathematical contract

| Item | Qwen contract |
|---|---|
| Backbone hidden width | 5120 |
| Native attention | 24 Q heads, 4 KV heads, head width 256 |
| Native activations/KV in this target representation | FP16 |
| Frozen selector projection | BF16 `[640,5120]`, row-major Linear weight |
| Selector heads | Four Q heads of width 128, one K head of width 128 |
| Learned selector norms | Q, token K, pooled K: three BF16 `[128]` weights |
| Norm scale | Zero-centered `1 + weight`, FP32 arithmetic, epsilon `1e-6` |
| Pool size / historical budget | Four tokens / 512 historical groups |
| Selector RoPE | First 64 coordinates only, rotate halves of 32 |
| Other selector coordinates | Last 64 remain unchanged |
| Native text RoPE | First 64/256 coordinates; actual native cos/sin values |
| Native nominal position range | 262,144; tested support must be stated separately |

Convert the target FP16 hidden input to BF16 for the frozen BF16 projection,
recording that boundary. Normalize each token K before pooling. Average four
normalized BF16 keys in FP32, cast the mean to BF16, apply pooled-K norm, then
rotate at the group's **start** position. Q normalization precedes its partial
rotation. Preserve all frozen parameter values; no fitting or recovery training.

For query position `t`, historical groups satisfy `g < floor(t/4)`. The score is
`sum_h ReLU(dot(selector_q[h], pooled_k[g])) / sqrt(128)` across the four
selector heads. Keep at most 512
historical groups, breaking threshold ties in favor of earlier group indices.
Emit retained historical groups in ascending order. Append the current block
exactly once, only through `t`, even when that block has four tokens. First actual
pruning is at query position 2052; maximum support is 2052 tokens.

Compute attention probabilities from the **native** Q/K dot products with the
native scale. Selector scores only choose support. Read selected positions from
the original full KV buffers. Any full-support fast path must update selector
cache state correctly and preserve the same later continuation semantics.

## State, ownership and supported requests

Persistent state belongs to each native node/session: pooled keys, incomplete
four-token tail, seen length, batch/capacity and native KV pointer identity.
Full prefill resets that state. Cached decode requires a matching completed
prefill and in-place KV buffers. Test all block remainders and replay after reset.

Use the EP compute stream and existing sequence-length readback where valid;
avoid adding a new host synchronization at each selected layer. Scratch used
only during the current operation can be shared within one serial session.
Large FP16-to-BF16 hidden/projection buffers need not remain separately live for
all four nodes. Retained pools and tails must remain distinct. This matters on
the 64 GiB shared-memory target.

The initial qualification uses unpadded sequential text, starting with batch one.
Canonical text position IDs and uniform lengths must be explicit in validation;
padding or custom multimodal positions must not silently use the text formula.
Expand batches 4/8 and 64K contexts after actual correctness and memory probes.
Record an implementation restriction separately from a hardware limitation or
the source AITER exclusion. A 65,536-token prefix plus 64 decode calls needs
capacity **65,600**, so do not inherit OLMo's 65,536 total-window cap.

## Numerical and quality checks

Before performance claims, validate projection and all three norms, token-key
normalization before pooling, block-start partial RoPE, stable selected indices,
current-block causality, GQA mapping, direct KV reads and output numerics.
Separate feature-projection rounding differences from ranking checks using
identical feature tensors. Include nonzero learned norm weights and threshold
ties. Validate cached continuation beyond 128 calls and all tail transitions.

The default four-block numerical controls already cover 1/7/17/513/4096-token
prefixes and a 4095+1 continuation against frozen dequantized HF weights. The
native port needs additional QSA-specific controls and full-model default/Changed
checks; those existing default diagnostics are not QSA validation.

CPU fixtures now match the canonical frozen source features bit-for-bit for all
four selector layers, using target FP16 hidden inputs cast to BF16 and actual
native FP32 RoPE tables. Stable ties, the first pruning query and causal support
counts are checked. `prepare_selector_fixtures.py` writes their tensors and hash
receipt under `_local/qwen-qsa/selector-fixtures-v1/`. They are CPU oracles for
later native tests; no native QSA qualification is implied.

Use the locked 64-window 4K test and 128-question zero-shot, thinking-disabled GSM
protocol. Source gates are PPL increase at most 5% and GSM drop at most 2 points
against the fresh default. OLMo's accepted nongating GSM decision is separate.
Natural long-document quality is locked before tuning and reported separately;
short GSM does not reach the pruning threshold.

## Timing and evidence

The frozen timing protocol has candidate prefixes 4K/16K/32K/64K, 64 common
teacher-forced decode inputs, two warmup pairs and six alternating measured pairs.
Time full-model prefill and decode independently with synchronization and the
same output-head/session/cache work. Sampling stays outside primary decode
timers. Preserve raw trials and report paired uncertainty and regressions.

Both models should be resident together where actual memory allows. If this is
infeasible, retain the failure and explicitly document any serial-loading timing
protocol; loading cannot enter one variant's measured phase. Do not infer a
speedup or feasible batch from sparse KV arithmetic or MI355X source rates.

Source trained job 303195 now times the same step-128 export as source accuracy.
Its greedy decode includes argmax and can follow different token streams, so it
remains a separate reference protocol from this native teacher-forced comparison.
