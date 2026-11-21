<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OLMo QSA inference port to Strix Halo

This page records the controlled-boundary prototype. The follow-on
[native GQA integration](olmo-qsa-native-gqa.md) removes its intermediate session
and stream handoffs; its measurements are reported separately.

## Status

The Original baseline is established with deterministic RTN W4A16 group128 and
the pinned model's per-layer rotary semantics. Frozen QSA is integrated through
four controlled attention boundaries and passes operator/cache/model validation.
The paired accuracy evaluation is complete: Dolma PPL rises 1.385% (passing the
5% gate), while GSM8K falls from 94/128 to 91/128, a 2.344 percentage-point drop.
The user accepted that pilot result and removed GSM8K as an acceptance gate.
All four paired timing suites are complete, using **batch one** throughout.
No training, fine-tuning, distillation, recovery, learned
calibration or selector updates are permitted or used. See the
[measured comparison](../experiments/olmo-qsa-results.md) for intervals and scope.

The boundary prototype improves decode throughput by 10.504% at 64K; decode
throughput falls at 4K/16K/32K, and prefill throughput falls at all four contexts.
Headline timing uses complete native Original and counts Changed's boundary and
orchestration costs. The 64K suite uses isolated workers after a combined-runtime
attempt reached the host-memory guard. Both variants fit and pass finite/exact
replay checks independently. Long-context quality remains unqualified: single
timing-stream continuations show larger PPL increases at 32K and 64K.

The final audit passed 128 Dolma rows, 256 GSM8K rows and 48 timing rows, verified
all 256 GSM8K prompts against the historical source, and checked immutable model,
kernel and selector hashes. Three GSM8K prompts initially decoded with Windows
CP1252 were corrected and both variants rerun; scores were unchanged. Earlier
receipts remain preserved. The audit receipt is `_local/qsa/final-audit.json`.

## Fixed comparison

| Component | Contract |
|---|---|
| Source | allenai/Olmo-3-1025-7B Base |
| Revision | a81bae42db3975be1671e27b9c9a56da1a9f980f |
| Chosen backbone | Deterministic RTN INT4, group128; hip-ep MatMulNBits; FP16 activations/KV; DP4A disabled |
| Original | All eight global layers retain dense attention |
| Changed | Only globals 7/11/15/23 use QSA; layer 19 remains dense |
| Selectors | Frozen step-4096 BF16 export from sparse-attn job 302354 |
| Export SHA256 | decd3160f4321839c80fee2ba0b0241575fc06017676e38a04286a3ae3daa10b |
| QSA budget | 128 complete groups of four, plus at most three causal tail tokens |
| Accumulation | FP32 attention and selector scores; 16-bit selector parameters |
| Projections | Identical Q/K/V/O and MLP quantization, weights and implementation in both variants |
| Rotary | Default RoPE for sliding layers, YaRN for globals; FP32 tables/arithmetic, then FP16 Q/K |
| Batch | One; no batched OLMo qualification is claimed |

The chosen W4A16 path explicitly sets `HIPDNN_EP_MATMUL_DP4A=0`; this project's
default enables W4A8 for small matmuls. An initial operator check with that
default produced relative L2 0.006543 against FP16 activations and was retained
as `operator-probe-default-dp4a.json`. With DP4A disabled, relative L2 is
0.000288 for decode and at most 0.000060 for tested prefill shapes.

## Established Original baseline

| Metric | Original | Scope |
|---|---:|---|
| Fresh-test PPL | 6.434832542 | 64 × 4096-token Dolma windows; 4095 targets each |
| Fresh cached NLL | 1.776615743 | Same documents, prefix4032 plus 63 calls predicting 64 targets |
| 4K prefill tokens/s | 1736.0084 | Initial complete-model B1 baseline; median of three timed requests |
| 4K decode tokens/s | 26.8346 | Initial complete-model B1 baseline; 64 fixed calls, independently timed |

The initial timing used two full exact-shape warmups. The output head is included
at the final position of every call. Tensor/binding creation and preallocated
cache creation are excluded. Decode excludes prefill/cache preparation and
sampling. Aggregate and per-request rates are equal at B1. Final paired timing
reruns the complete native Original against integrated Changed, including all
Changed boundary/orchestration costs. The initial three-trial median is retained
as the baseline record, not substituted for the balanced paired control.

| Corrected-model check | Relative L2 | Result |
|---|---:|---|
| Complete 4K Original versus Transformers 5.19.0, same RTN weights | 0.000387408 | Pass |
| Complete Original, full versus cached 4092+4 | 0.000246872 | Pass |
| Complete Original replay | 0, bit exact | Pass |
| Partitioned dense Original versus complete Original, tested prefill/decode | 0, bit exact | Pass |
| Integrated Changed, full versus cached 4092+4 | 0.000319364 | Pass |
| Integrated Changed 517+4 replay | 0, bit exact | Pass |

Fresh quality and timing receipts are in `_local/qsa/baseline-w4-rope-v2/`.
The accepted graph is `_local/qsa/original-w4-rope-v2/original.onnx`, SHA256
`1efc13593017cba9899d8aa6407d6912c16dcc719cfef480c9aed9b97e3ac741`.
Its constants SHA256 is
`04f0f1c70dffb795b3a243be7a15886aaa822e35e9486cc9a8afa5cd148df999`.

## Controlled attention boundary and runtime

`scripts/qsa/partition_original.py` exports five backbone graphs alternating with
four dense attention-middle graphs at 7/11/15/23. All 224 MatMulNBits nodes are
byte-identical to the established Original graph. The accuracy/control runner uses
the same five compiled backbone session objects, packed constants, Q/K norms,
FP32 RoPE, embeddings and output head. Its Original calls native GQA middles; Changed
calls the QSA C ABI at those boundaries. Layer 19 and 27 other attention layers
remain in the common backbone graphs.

This is an experimental Python-controlled inference boundary with an explicit
partition manifest, not a new monolithic ONNX custom-op lowering. The existing
EP and custom-kernel binaries are used unchanged for both variants. The staged
QSA library is built separately from
`lib/Runtime/Kernels/qsa/CMakeLists.txt` using the project's per-architecture
kernel helper. Its pure-C contract is
`lib/Runtime/Kernels/include/hip_qsa.h`; device implementation is
`lib/Runtime/Kernels/hip/qsa/qsa_port.hip`.

The native pipeline appends new FP16 KV, constructs BF16 selector query/pooled-key
features, computes FP32 scores and causal exact radix top-k, then performs
FP32 selected-address attention and partition merging to FP16 output. BF16
projection uses Torch GEMM with a persistent output buffer. Current FP16 hidden
states are copied/cast into a persistent BF16 selector input buffer. Selector
parameters remain **unchanged BF16**; no FP16 parameter conversion is needed.
Default/YaRN rotary tables and arithmetic remain FP32. No full-history KV gather,
cast or copy occurs inside QSA. Full KV remains stored for every layer; this is
not KV compression. All QSA workspaces and cache pointers persist through decode.

The noncooperative launch sequence uses wave32 reductions and stream ordering.
It does not claim one fused kernel or a fused whole-model/prefill implementation.
Headline timing keeps Original as a single complete native graph. Changed's
explicit stream handovers and orchestration are included in its measured time.
Both use the same native projection kernels, quantization, physical GPU and
shared persistent KV buffers. A preliminary timing diagnostic that partitioned
both variants is retained under `_local/qsa/boundary-performance-diagnostic/`,
and excluded from the headline comparison. Decode preserves full prefill followed by sequential one-token
cache semantics; padding, beam search, cache rollback and continuous batching
are outside this experiment.

The 200 raw cases cover all four frozen selectors, ten prefix lengths from 1 to
4097, and four subsequent decode steps each. Exact cache append/suffix checks,
causal bounds, selected-group uniqueness, FP32 top-k optimality, deterministic
ties and selected-address attention passed. Maximum relative L2 was 0.001130 for
query features, 0.0000582 for pooled features and 0.0000361 for attention.
Top-k optimality gap was zero in sampled checks. Integrated replay/finite checks
and selector before/after integrity passed. Raw evidence:
`_local/qsa/qsa-raw-validation.json`, `changed-validation.json` and
`original-w4-rope-v2/partitioned/validation.json`.

Paired quality outputs are `_local/qsa/paired-quality/`; final phase timings are
`_local/qsa/paired-performance/`. Results are pending and are not inferred from
the MI355X measurements.

## Reproduction entry points

Use `_local/qsa/venv/Scripts/python.exe`, with Transformers 5.19.0, ONNX 1.19.1
and ORT 1.31.0. This task-local environment reuses the installed ROCm Torch
2.13.0 stack and read-only ONNX dependency paths. `QSA_HIP_EP_INSTALL` can select
an equivalent install, but every comparison must verify and record its actual
EP/kernel hashes. The default is the verified existing install described below.

The source scripts execute in this order:

1. `prepare_baseline.py` / bounded `download_ranges.py`: pinned public model and
   SHA256 verification. Existing source and checkpoint directories are read-only.
2. `export_original.py`: deterministic RTN and per-layer FP32 rotary tables.
   Existing exports are protected from overwrite. A fresh export directly emits
   FP32 RotaryEmbedding nodes; `repair_rope_graph.py` is the retained one-time
   mixed-dtype prototype repair and must not be applied to a fresh graph.
3. `original_inference.py --source-reference --length 4096`, then
   `benchmark_original.py` and `evaluate_original.py dolma --cached` establish
   the complete-model Original baseline.
4. `partition_original.py`, `build_qsa.py`, `validate_partitioned.py`,
   `validate_qsa.py`, `validate_changed.py`: controlled export, native build,
   dense equivalence, raw numerical/cache checks and integrated validation.
5. `evaluate_pair.py dolma`, `evaluate_pair.py gsm8k`,
   `validate_qsa_long.py`, and `benchmark_pair.py --context N --pairs 6` for
   `N=4096,16384,32768,65472`: paired accuracy and independently timed phases.
6. `report_results.py`: audit complete raw rows and generate the
   [measured results](../experiments/olmo-qsa-results.md).

Wrap stages with `run_stage.py LABEL -- SCRIPT ARGS` to snapshot sources and
commands and serialize GPU work. Use `run_stage.py --cpu LABEL -- SCRIPT ARGS`
only for CPU-only download/export/build/report stages. Labels are unique and
partial failures remain in their original stage directories. The wrapper checks
selector SHA256 before/after and bounds each stage to one hour, stopping only its
own process tree on timeout. Existing raw timing rows are never overwritten.

## Invalid prototype numerical evidence

| Check | Relative L2 | Cosine | Result |
|---|---:|---:|---|
| Full 32-layer model, cached 517+4 versus full prefill | 0.0003463 | 1.000000 | Pass |
| Full model versus pinned Transformers, identical RTN weights | 0.0005300 | 1.000000 | Pass |
| Full model prefill replay | 0 (bit exact) | 1.000000 | Pass |
| Global/sliding MHA cache append | 0 for tested K/V writes | — | Pass |

Operator tests include past lengths 0, 17 and 4097, the 4096-token sliding
boundary, aliasing of past/present storage and untouched cache suffixes. These
are numerical checks, not model-quality results. The one-layer diagnostic's
first reference attempt failed because its meta-device rotary buffer had not
been materialized; that failed receipt is retained alongside the passing retry.
The full 32-layer reference passed on its first attempt.

Both the corrected experiment and the prototype use the existing gfx1151 install
from `_local/ring-1259/install/existing`, not a claim that this checkout's current
HEAD was built. EP SHA256 is
`00a220e560cae4f625f926772e422e2700ba8bed383a3f5e0f2e25c9c794d875`;
kernel library SHA256 is
`2cf0c580b612024834967e78c55809e37a35fa5afa6b5dfa70f39553ea3c6ace`.
The matching saved build receipt records source HEAD
`e99e1ae046a9dda10417a95135e46e750bc8f3d8`, upstream
`65d1f5a276402d1ab33474f268af9350791a3cba`, with both binary hashes above.
It is `_local/ring-1259/results/build-existing.json`; the recipe/log remain
beside it. The campaign's source checkout has since moved to `d4633808`, so its
current HEAD must not be substituted for the build-time source. Any final
paired comparison must use the same library pair for both variants.

## Initial Original quality investigation

| Run | Documents | Prefill PPL | Cached NLL | Interpretation |
|---|---:|---:|---:|---|
| Invalid all-YaRN prototype, RTN INT4 group128 | 64 × 4096 | 24.545209 | 4.344428 | Completed; wrong sliding-layer rotary semantics; excluded |
| MI355X MXFP4 source reference | Same 64 | 6.671988 | 1.824839 | Different quantizer/device; not a hip-ep baseline |

The 4K standalone INT4 matmul and dense/sliding MHA tests also pass (relative L2
0.0000597 and 0.0000849 respectively). These random-input operator tests do not
prove model semantics. A full 4096-token Transformers 5.5.3 comparison with
identical RTN weights reproduced the error (first-document PPL 19.126938 versus
hip-ep 19.123842). Plain FP16 under that same wrong configuration still gave
PPL 17.848255, disproving quantization as the main explanation. The source packed
manifest explicitly records per-layer-type rotary parameters. Corrected exports
must use those semantics, independently assert both table types and match the
5.19.0 reference. No learned calibration or recovery is allowed.

Raw receipts are `original-baseline/dolma.jsonl`,
`original-baseline/dolma-summary.json`, `operator-probe-large-prefill.json` and
`quantization-diagnostic/`, relative to `_local/qsa`. Subsequent stages snapshot
all Python sources, exact commands, exit codes and selector before/after hashes
under `_local/qsa/stages/`.

## Hardware receipt

Windows 11 build 26200; Ryzen AI MAX+ PRO 395; Radeon 8060S gfx1151, 20 reported HIP multiprocessors,
wave32; graphics driver 32.0.22050.4002. HIP runtime/driver 71526333;
Torch 2.13.0+rocm10.0.0 (HIP 7.15.26333); ORT 1.31.0; Transformers 5.19.0
in the task-local environment (5.5.3 only in the excluded prototypes);
Python 3.13.13. Installed RAM is 128 GiB, with 63.776 GiB managed by Windows after
firmware reservation. HIP reports a larger GPU-addressable range; that value is
not a safe allocation budget by itself.

Both cooperative-launch device attributes return zero. The MI355X cooperative
kernel cannot run here; QSA requires a staged launch design with explicit stream
ordering and wave32 reductions. Full hardware and library hashes are retained in
`_local/qsa/environment.json`.

The HIP compiler is 7.15.26333-6b0e43f341, AMD clang 23.0.0git at
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`. The staged library's build command,
source hashes and binary digest are in `_local/qsa/native/build.json`.

## Execution order and evidence

1. Verify pinned model/source/selector and available hardware/runtime. Export
   supported quantization without calibration or optimization updates. Establish
   strict GPU Original inference, numerical/cache checks, and baseline quality.
2. Integrate a controlled attention-middle boundary with frozen selectors and
   explicit cache/workspace ownership. Validate operator, prefill/decode and
   boundary semantics before model evaluation.
3. Evaluate Original versus best Changed on the same device, backbone, inputs,
   output-head work and cache semantics. Time prefill/decode independently using
   exact-shape warmups and balanced synchronized repeated pairs. Decode excludes
   cache preparation/prefill and sampling. Report aggregate and per-request rates.

Use the source project's 64 held-out Dolma documents of 4096 tokens and reused
128-item GSM8K pilot/parser where possible. Assess QSA relative to hip-ep Original,
not the MI355X MXFP4 model. The acceptance gate is at most 5% fresh-test PPL
increase. GSM8K is diagnostic: the user accepted its observed 2.344-point drop
and removed the original 2-point gate. These do not establish broad long-context quality.
Keep batch one and expand contexts to 16K/32K/64K after shape correctness and
memory checks. Retain commands, raw rows, hashes, versions and failures under
`_local/qsa/`; durable summaries belong here with README links.

## Source references

Read-only sibling `sparse-attn` provides README, PROJECT_STATUS, ARCHITECTURE,
EXTENDED_RESULTS, PERFORMANCE, DATASETS and REPRODUCIBILITY, and
`scripts/hip/qsa_fused.cu`, `fused_decode.py`, `fast_selector.py`, `append_cache.py`,
`trained_performance.py`. Its running performance continuation is owned by the
parent chat and must not be managed here. Existing hip-ep edits and experiments
are preserved. No push or public publication is authorized.
