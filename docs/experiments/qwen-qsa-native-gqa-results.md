<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen trained-selector port: native default and QSA

## Current evidence

Fresh 64-block native default quality is complete and independently audited. Four-block architecture diagnostics pass native HIP and frozen Hugging Face numerical controls. Native QSA integration passes kernel, complete-operator and full-model numerical/dispatch controls. Matched 4K PPL/GSM results are below; long-document QSA accuracy and performance remain separate qualifications.

Work is isolated on `codex/qwen-native-qsa`, created after OLMo accuracy report commit `7e0e1e3`. The earlier OLMo branch, binaries and results are preserved. The source sparse-attn project is read-only. Training, fine-tuning, recovery training, distillation, learned calibration and selector updates are zero.

## Frozen identities

| Input | Identity |
|---|---|
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Trained selector payload | `d7729599f421d80ace536f84145bafa9a6e1eaad22dde18ce42392014a58b5e4` |
| Selector checkpoint | Job 303042, step 128; layers 27/31/35/39 |
| Selector tensors | 16 BF16 tensors, 13,108,736 parameters; every shape and raw-byte hash verified |
| Backbone | All 18 official BF16 shard byte hashes match the source identity manifest |
| Full text backbone | 26,895,998,464 parameters; 53,791,996,928 BF16 bytes, excluding vision/MTP |

The official local snapshot and selector/data payloads remain unchanged. No OLMo or community selector is substituted. Source MI355X results are references: BF16 PPL 5.787418 → 5.791103 and GSM 109/128 → 111/128. They are not hip-ep measurements or a substitute for a fresh converted default.

## Target representation and native behavior

The target is **AMD Radeon(TM) 8060S Graphics**, gfx1151/wave32, with 63.78 GiB physical shared memory. The native probe uses PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333 and Transformers 5.19.0. The default measurements use `_local/qwen-qsa/install`, which preserves the unchanged four hash-verified OLMo native compiler/runtime binaries. The Qwen port uses a separate build/install directory.

Native GQA currently supports FP16/FP32 rather than BF16. Full text FP16 weights alone occupy about 50.10 GiB before GPU cache/workspaces and host/compiler copies. A fresh deterministic RTN W4A16 group128 backbone is therefore selected for this machine; small Gated DeltaNet a/b projections, embedding and output head remain FP16. Both later variants must use these exact exported backbone bytes. This is a new conversion from the official checkpoint, with its own default accuracy; it is not a restart of Qwen MXFP4 or reuse of earlier quantized accuracy claims.

The pinned release ONNX supplies topology and static RoPE/shape constants only. Every learned constant is regenerated from the verified official BF16 weights. Its topology hash is `44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd`. Export manifests record exact source-parameter coverage, per-projection conversion errors, representation, output graph/data hashes and exporter hash. The four-block diagnostic contains 5,940,028,480 bytes of fresh constants. The full export has 64 blocks: 48 Gated DeltaNet and 16 full-attention layers; native attention is Q24/KV4/D256.

The current LinearAttention ABI has one dtype for Q/K/V, gates and recurrent state. Explicit FP32 LinearAttention regions and casts retain **FP32 recurrent state** with FP16 backbone boundaries. The rest of the graph remains identical between the measured variants. Its conversion and numerical error must be evaluated rather than called BF16-exact.

**Gate metadata control:** the checkpoint config includes `output_gate_type=swish`, but the actual Transformers 5.19 Qwen3_5Attention class used by the source experiment executes sigmoid gating. A CPU single-key control matches sigmoid exactly (maximum absolute error 0) and differs from SWISH by 1.761594. The export preserves the executed native **sigmoid** behavior; changing it based solely on that extra metadata field would change the source model.

## Native default numerical controls

The four-block diagnostic contains three Gated DeltaNet blocks and one full-attention block, using real checkpoint weights. Native sessions have CPU fallback disabled. It checks finite prefill/decode outputs, exact replay after state reset and FP32 recurrent buffers.

Full eight-token prefill versus seven-token prefill plus one decode differs by 0.04850% relative RMS. Replay is bit-exact.

The independent HF control uses the exact unpacked RTN weights and FP16 unquantized parameters, with no fitting. The predeclared diagnostic limit is 2% relative RMS.

| Prefix / continuation | Native vs frozen HF relative RMS |
|---|---:|
| 1 tokens | 0.02960% |
| 7 tokens | 0.03740% |
| 17 tokens | 0.03881% |
| 16 + 1 cached token | 0.03246% |
| 513 tokens | 0.02997% |
| 4096 tokens | 0.03799% |
| 4095 + 1 cached token | 0.03745% |

These are architecture/export diagnostics, not a full-model accuracy score or performance benchmark. The HF fallback implementation is used for correctness only; no timing comparison against that fallback is claimed.

## Fresh native quality

The locked Qwen protocol uses 64 test windows of 4,096 tokens and 128 zero-shot, thinking-disabled GSM questions, greedy generation with a 512-token reserve and both official EOS IDs. The parser takes the last #### number, falls back to the last number, and applies Decimal normalization. It does not use OLMo's five-shot prompt/strict parser. Qwen gates remain PPL increase ≤5% and GSM drop ≤2 percentage points against the fresh default. The earlier OLMo nongating decision is not transferred to Qwen.

| Metric | Fresh native default |
|---|---:|
| PPL, 64 × 4K windows | 5.968688 |
| GSM8K, 128 questions | 111/128 (86.72%) |

All rows pass protocol/data identity and score/count audits. 18 default GSM generations reached the cap. Raw completions and capped cases are retained. The separate matched default/QSA results follow below.

## Native Qwen integration controls

The isolated Qwen native build passes 611 supported compiler tests (nine unsupported) and three compiler/runtime unit tests. Frozen GPU features agree with the canonical CPU fixture within 0.004% relative RMS; tested block IDs and stable ties match exactly. Incremental pools match full prefill exactly. Dedicated batch-four/eight feature and long-index controls qualify those kernels, not full-model batch/long-context feasibility.

The complete native operator, including hipBLASLt BF16 projection and dense-to-sparse cache transition, matches the independent frozen CPU attention oracle within 0.030% relative RMS. Full-model Original/Changed outputs are bit-exact on the tested full-support requests (prefixes 1/17/513/2047 plus four decodes). Sparse 4092+4 versus full prefill differs by 0.07771% relative RMS and replays exactly; first-pruning continuity differs by 0.10323%. ORT profiles verify one EP node dispatch per forward: 20 Original and 39 Changed calls, with unchanged cache addresses.

The validated implementation is committed as `7a71b72` on `codex/qwen-native-qsa`. Both timing variants use the isolated Qwen build and identical backbone bytes. The earlier fresh default remains preserved. Matched quality is independently evaluated below; numerical controls do not substitute for dataset accuracy.

| Metric | Matched native default | Matched native QSA |
|---|---:|---:|
| PPL, 64 × 4K | 5.968688 | 5.971270 |
| GSM8K, 128 questions | 111/128 (86.72%) | 111/128 (86.72%) |

**Matched quality gates pass:** PPL changes by +0.043265% against the 5% limit; GSM changes by +0.00 percentage points against the 2-point drop limit. Default/QSA retain 18/18 capped generations. Generated GSM token sequences differ on 0/128 questions. This GSM pilot stays entirely within the native full-support dense path; it does not measure quality after sparse pruning. The 4K PPL test exercises sparse pruning.

## Matched native performance

Full-session wall timings use two exact-shape warmup pairs and six balanced measured pairs, with a common 64-token teacher-forced continuation. Both variants include the same final-position output head and resident cache work. Model loading, input/output/binding allocation, recurrent-state zeroing and sampling are outside phase timers. Intervals are 10,000 paired bootstrap resamples and describe within-run variation.

| Batch | Prefix | Phase | Default tok/s | QSA tok/s | Change, paired 95% interval |
|---:|---:|---|---:|---:|---:|
| 1 | 4K | prefill | 438.06 | 419.97 | -4.13% [-4.34, -3.75] |
| 1 | 4K | decode | 10.68 | 10.74 | +0.50% [+0.14, +0.92] |
| 1 | 16K | prefill | 415.94 | 397.34 | -4.47% [-4.57, -4.40] |
| 1 | 16K | decode | 10.13 | 10.25 | +1.23% [+0.66, +1.80] |
| 1 | 32K | prefill | 384.85 | 350.87 | -8.83% [-9.18, -8.43] |
| 1 | 32K | decode | 9.36 | 9.51 | +1.60% [+0.22, +2.99] |

Every listed row has finite outputs, exact warmup replay, unchanged sessions/cache addresses, and independently audited paired raw trials and statistics. Shapes without a completed receipt remain unqualified.

**64K has no valid timing result.** The co-resident two-session attempt stopped during warmup when OS available RAM remained below the 2 GiB guard. Its memory trace and failure receipt are retained. Further 64K work was excluded at the user's request. The [separate default/QSA memory report](qwen-qsa-memory.md) selects batch1/4K as the memory comparison: 19.3376 versus 19.5849 GiB peak WDDM total commitment. The completed batch4/4K pair remains supplementary. All remaining memory probes were canceled by the user; no memory runs are queued. Each completed probe used one prefill and eight decode calls in an isolated process, without performance repetitions.

## Earlier paired-run application memory

A separate CPU-only sampler reads OS process/system and Windows PDH GPU memory gauges every second without creating a GPU context. The worker holds Original and Changed sessions together, sharing KV serially; these are joint application footprints, not isolated per-variant allocations. The 32K trace attaches mid-run; 64K sampling starts before application startup. Sampled peaks can miss sub-second spikes.

| Prefix | Coverage | Peak process RSS GiB | Peak private commitment GiB | Peak WDDM dedicated usage GiB | Peak WDDM shared usage GiB | Minimum system RAM available GiB |
|---|---|---:|---:|---:|---:|---:|
| 32K | partial, attached mid-run | 6.35 | 65.71 | 61.67 | 5.49 | 30.89 |
| 64K | startup through failed attempt | 39.06 | 96.59 | 61.70 | 38.19 | 0.08 |

WDDM values are driver-reported usage/commitment; they are not a proven physical-residency total. Process RSS, private commitment, GPU dedicated/shared gauges and system RAM overlap and must not be added. Raw timestamped samples, OS lifetime high-water counters, per-request completion markers and immutable source/snapshot identities are retained under `_local/qwen-qsa/memory-v1/`. Memory sampling begins after the earlier uninstrumented trials and partway through 32K; 64K has the observer throughout both variants.

## Separate locked long-context inputs

Before native QSA integration or kernel tuning, four natural documents from the pinned Dolma PDF subcorpus were selected with seed 20261124, one per new shard. Prior Qwen selector-data shards and all train/validation/test document hashes are excluded. The bounded selection also rejects shared 16-token fingerprints with the 128 GSM questions. This is not a semantic contamination guarantee or a corpus-representative sample.

Each document supplies nested 16K and 32K prefix trials, scoring the final 1,023 next-token targets at each length. The four locked windows contain 32,896 tokens each, including a 128-token reserve. The document file SHA256 is `a91f55b001e2e345072ee9da3f329e90486739e4d405e21d3cf889345889b3a4`. These suffix scores are reported separately from the 4K and GSM diagnostics.

The timing inputs are separately frozen under `_local/qwen-qsa/timing-v1/`, SHA256 `5c8ac2934c8f90d049337e50ea89b95a0c427c1b9b21291cc9d18de1827e567b`. Eight common 65,600-token streams concatenate the earlier locked 4K test windows. They support candidate prefix lengths 4K/16K/32K/64K with 64 teacher-forced decode calls, and candidate batches 1/4/8. The protocol declares two warmup pairs and six measured alternating pairs. These are candidate shapes rather than claims of memory/support, and these concatenations are timing inputs rather than natural long-document quality evidence.

| Natural-document prefix | Default suffix PPL | Native trained QSA |
|---|---:|---|
| 16K, four documents / 4,092 targets | 6.409937 | Unmeasured |
| 32K, four documents / 4,092 targets | 5.942933 | Unmeasured |

All listed long-context default outputs were finite and pass document, protocol, row-count and target-count audits. This four-document diagnostic is not a corpus-wide accuracy claim.

## Updated MI355X trained reference

Source job 303195 now times the exact trained step-128 export used for source accuracy, with an unchanged backbone and six unchanged inference/kernel sources. Source packet commit is `3b0c30ea4976085798a36d22cefa4089bbef6543`. This evidence update does not change the transferred selector or native quality protocol. Selected batch-one median paired changes are:

| Source prefix | Source decode TPS change | Source prefill TPS change |
|---|---:|---:|
| 16K | -1.24% | -1.64% |
| 32K | +11.93% | +0.00% |
| 64K | +17.73% | +2.26% |

These are BF16 MI355X results, not hip-ep measurements or a forecast. The source has two warmup pairs and six measured alternating pairs per shape. Its independent wall phase timers include per-step greedy argmax in decode, and variants can follow different token streams. That protocol remains distinct from the completed native fixed-token timing. Full source tables, raw-trial linkage and ranges are in the read-only `../sparse-attn/QWEN_PERFORMANCE.md` and `docs/reproducibility/qwen-trained-performance.json`. Historical untrained timing remains separate.

## Unmeasured coverage and current status

The native wave32 integration and feature/index/cache/numerical controls above are complete. Only layers 27/31/35/39 are replaced, using the frozen Qwen projection, three zero-centered norms, token-key normalization before four-token FP32 pooling, partial 64-coordinate RoPE, stable 512 historical-block selection and one causal current block. Full KV storage, selected-address reads and all Gated DeltaNet/default work are preserved. The matched 4K PPL/GSM campaign is complete; long-document quality is qualified only where completed matched rows appear above.

Prefill and decode use independent fixed-token phase timers, identical session/head/cache work, exact-shape warmups and balanced paired trials. Only completed, audited shapes in the performance table establish measured throughput or support; remaining batch/long-context candidates require actual memory and correctness checks.

The [paired50-case BFCLv4 component pilot](qwen-qsa-bfcl-v4-50.md) is complete: default26/50 versus QSA24/50. Prompt/response lengths, turns, cap counts and actual pruning coverage accompany the scores. This is not the official overall BFCLv4 score. No further hip-ep benchmarks are scheduled. The official step bound stops2/3 default/QSA cases; per-turn token/call records are retained. Matched QSA natural-long quality and Qwen SWE-bench remain unmeasured. The local merged PR candidate has separate correctness checks and no new performance/accuracy campaign. See the [audit and paper handoff](../reproducibility/qwen-hip-ep/README.md) for its validation receipts, the selected memory pair and complete raw evidence. The branch and results remain local; nothing has been pushed or published.

## Evidence and reproduction

Scripts are under `scripts/qwen_qsa/`. CPU identities/support receipts and export/conversion manifests are under `_local/qwen-qsa/`. Bounded stages use the same serial GPU lock as OLMo, with retained command/status logs under `_local/qsa/stages/qwen-*`. Four-block support and numerical receipts retain binary, graph, validator and runtime identities. `report_results.py` audits complete quality rows before reporting a full default score.

The [native Qwen GQA design contract](../design/qwen-qsa-native-gqa.md) records the replacement boundary, precision, cache ownership, selection semantics and validation/timing requirements before implementation.

See the preserved [OLMo native accuracy report](olmo-qsa-agent-accuracy.md) and [OLMo native performance report](olmo-qsa-native-gqa-results.md). GSM/BFCL pilots are complete; SWE-bench remains blocked by the missing container host.
