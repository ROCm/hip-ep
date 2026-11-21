<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native QSA specialization of GQA

The user approved integrating frozen QSA at the GQA operator level to remove
the boundary prototype's repeated ONNX-session calls and Torch/HIP handoffs.
`com.amd::QsaGroupQueryAttention` lowers to `hip.qsa_gqa` and one native runtime
wrapper. Four nodes replace global attention middles 7/11/15/23 in the complete
OLMo graph. Layer 19 and all other attention layers retain native dense/sliding
GQA. All 224 quantized backbone projection nodes remain byte-identical.

Both Original and Changed execute as complete ONNX graphs with one session run
per forward. QSA stages enqueue on the EP-owned HIP stream. There is no Python
selector projection, ctypes attention call, intermediate ORT session, or
device-wide barrier at an attention boundary. Internal GPU stages still perform
the selector, selection and sparse-attention work; this is not a single fused
GPU kernel. Dispatch count is checked with ORT profiling in numerical validation.

## Operator contract

The specialization is deliberately narrow: gfx1151 wave32, batch one, 32 query
and KV heads, head dimension 128, causal attention, at most 65,536 total tokens.
Q/K normalization and backbone FP32 RoPE remain in the surrounding graph.
Selectors remain unchanged BF16 from the step-4096 export. Training,
fine-tuning, recovery, distillation and learned calibration are all zero.

| Input indices | Values | Dtype / layout |
|---|---|---|
| 0–2 | Rotated Q, rotated K, V | FP16 `[1,S,4096]` |
| 3–4 | Past K/V storage | FP16 `[1,32,C,128]` |
| 5–6 | `seqlens_k`, capacity | int32 `[1]`, scalar; standard GQA convention |
| 7 | Layer input hidden state | FP16 `[1,S,4096]` |
| 8 | Frozen selector projection | BF16 `[640,4096]` |
| 9–10 | Frozen selector Q/K norm weights | BF16 `[128]` each |
| 11–12 | Selector global cosine/sine tables | FP32 `[65536,64]` each |
| Outputs 0–2 | Attention output, present K/V | Same shapes/types as inputs 0/3/4 |

Only full prefill followed by one-token cached decode is admitted. Decode must
use the same in-place K/V storage, capacity and contiguous history as the prior
prefill on that node. Cache rollback, beam search, padding, multiple simultaneous
histories in one session and continuous batching are outside this contract.
Other models and geometries continue using the existing GQA implementation.

The BF16 ONNX importer mapping is explicit, including external initializers.
The dtype-mapping unit test checks exact BF16 raw bytes. The compiler rejects
FP16 substitutes for the BF16 selector parameters and incompatible Q/K/V/KV
geometry instead of interpreting equal-width storage as the wrong dtype.

## State and execution

Each native node owns an `OpState` slot containing selector pooled-key history,
partial-group tail, persistent query/score/selection/attention scratch and
prefill hipBLASLt descriptors. Session teardown releases this state. Independent
sessions own independent histories and workspaces.

Decode projects the current hidden state with a native BF16 GEMV, rounding
FP16 input to BF16, accumulating in FP32 and writing BF16 output. Prefill casts
the current hidden block and uses a BF16 hipBLASLt GEMM with FP32 accumulation.
No selector parameter is converted to FP16. Workspace grows only when needed
and is reused by subsequent decode calls.

The existing staged QSA kernels append new KV and selector features, select
128 complete four-token groups with deterministic lower-ID ties, and attend
directly at selected addresses with FP32 accumulation. There is no full-history
KV gather, cast or copy for sparse attention. Full KV storage remains allocated.
The validated FP-contraction setting applies only to QSA's kernel translation
unit; existing backbone kernel flags remain unchanged.

The native wrapper reuses GQA's per-Compute sequence-length cache. In the OLMo
graph, preceding dense layers populate that cache, avoiding an additional host
readback or synchronization at QSA nodes. Runtime failures also set the existing
EP device error flag consumed by the generated inference interface.
The EP propagates a nonzero inference status through its Compute callback as an
ORT execution error. Validation rejects decode without prior prefill and checks
that a subsequent full prefill clears the error and rebuilds valid node state.

## Reproduction and evidence

Use `scripts/qsa/build_native_gqa.py` for the isolated local installation and
`export_native_gqa.py` for matched B1 Original/Changed graphs. Local artifacts
live in `_local/qsa/native-gqa/`; the source model and selector hashes retain the
contracts in the [original port audit](olmo-qsa-port.md).

`validate_native_gqa.py` compares complete dense Original with its partitioned
control and complete native QSA with the frozen boundary reference. It checks
prefill/cached numerics, exact replay, cache pointers and one profiled EP node
dispatch per forward. `benchmark_native_gqa.py` independently times prefill and
64 fixed decode calls with two exact-shape warmups and balanced repeated pairs.
Its `--dolma` mode rechecks the retained 5% PPL gate on the same 64 held-out
document windows. GSM8K remains diagnostic, as accepted by the user.

The [boundary prototype results](../experiments/olmo-qsa-results.md) describe
the earlier multi-session implementation and must not be presented as native
GQA measurements. Native validation and measurements are required before making
performance claims for this path.
