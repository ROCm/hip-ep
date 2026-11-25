<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen eight-selector BFCL v4: fixed 50-case pilot

**Eight: 29/50 (58.0%)**, versus reused Original **26/50 (52.0%)** and historical four-selector **24/50**. Delta versus Original is **+6.0 percentage points**; paired stratified95% interval **[-2.0, 14.0]pp** (10,000 paired case resamples within category,seed42). This is an unweighted component pilot,not the official BFCLv4 overall score; it is nongating and does not establish population equivalence.

Only Eight was generated in this campaign. Original scores/outputs are unchanged and reused from the prior audited native campaign. The identical fixed seed42 manifest contains26 single-turn cases (two from each of13 categories) and24 multi-turn cases (six from each of four categories); web-search,memory and format-sensitivity are excluded. Official prompting/parsers/simulators/checkers at commit`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` are unchanged.

The model is Qwen/Qwen3.8-27B,native RTN W4A16 group128,FP16 activation/KV,FP32 recurrent state. All eight BF16 selectors at zero-based layers3/7/23/27/31/35/39/63 use checkpoint640 weights. Sparse QSA is enabled for prefill and decode; one resident full-model native session and one EP dispatch per forward. No training,calibration or kernel rebuild. Greedy generation uses the actual Qwen chat template,thinking disabled,both official EOS IDs,cap1024 per response,capacity32768. Official step-limit20 semantics remain unchanged (a stopped turn can execute21 calls). No stopped/capped cases are replaced and no output-format repair is applied.

## Measured workload

| Metric | Reused Original | Eight |
|---|---:|---:|
| Model calls | 294 | 295 |
| Response-token caps | 0 | 0 |
| Reduced context budgets | 0 | 0 |
| Official step-limited cases | 2 | 1 |
| Actual QSA-pruned cases | 0 | 24 |
| Prompt tokens/call,min / median / p95 / max | 431 / 7417.5 / 11977 / 17407 | 431 / 7481 / 12886.8 / 15913 |
| Completion tokens/call,min / median / p95 / max | 4 / 23 / 163.7 / 749 | 4 / 25 / 161 / 749 |
| Declared turns/case,min / median / p95 / max | 1 / 1 / 6 / 7 | 1 / 1 / 6 / 7 |
| Observed turns/case,min / median / p95 / max | 1 / 1 / 6 / 7 | 1 / 1 / 6 / 7 |
| Calls/case,min / median / p95 / max | 1 / 1 / 15.55 / 33 | 1 / 1 / 17.55 / 22 |

Prompt totals include repeatedly processed chat templates,tool schemas,history and tool results. Completions include a generated EOS. A turn is a benchmark user round; each turn can contain many model/tool calls. Actual calls and histories may differ between variants,so elapsed evaluation time is not a fixed-work speed comparison.

## Frozen initial-prompt groups

| Group | Cases | Original correct | Eight correct |
|---|---:|---:|---:|
| Initial prefill reaches pruning | 24 | 6 | 9 |
| Initial prefill stays within full support | 26 | 20 | 20 |

The grouping is frozen before generation; later-history pruning coverage is retained separately. First pruning occurs at query position2052 (zero-based). Every new prompt was independently retokenized,every completion independently decoded,and all50 new rows independently rescored with official checkers. Original retains its prior audit; protocol/queue/source hashes bind reused and new evidence separately.

## Component scores

| Component | Cases | Original correct | Eight correct |
|---|---:|---:|---:|
| `simple_python` | 2 | 1 | 1 |
| `simple_java` | 2 | 2 | 2 |
| `simple_javascript` | 2 | 2 | 2 |
| `multiple` | 2 | 2 | 2 |
| `parallel` | 2 | 1 | 1 |
| `parallel_multiple` | 2 | 2 | 2 |
| `irrelevance` | 2 | 1 | 1 |
| `live_simple` | 2 | 2 | 2 |
| `live_multiple` | 2 | 1 | 1 |
| `live_parallel` | 2 | 1 | 1 |
| `live_parallel_multiple` | 2 | 2 | 2 |
| `live_irrelevance` | 2 | 2 | 2 |
| `live_relevance` | 2 | 1 | 1 |
| `multi_turn_base` | 6 | 3 | 1 |
| `multi_turn_miss_func` | 6 | 1 | 2 |
| `multi_turn_miss_param` | 6 | 1 | 3 |
| `multi_turn_long_context` | 6 | 1 | 3 |

**The source checkpoint still misses its fixed32K PPL gate.** This task pilot does not establish accepted long-context quality. Eight-selector GSM8K and application-memory comparisons remain unmeasured.

[Paper handoff](../reproducibility/qwen-eight-hip-ep/README.md) · [Summary JSON](../reproducibility/qwen-eight-hip-ep/bfcl-eight-summary.json). The packet's`raw/bfcl-eight-v4-50/` contains per-case,per-generation and per-turn CSVs,official/token audit receipts,and lossless byte-chunked combined row records. The local ZIP also retains complete request messages,rendered prompts,token IDs,all source snapshots,attempt/recovery logs,and referenced Original raw requests. Infrastructure failures are preserved separately and are never scored as wrong answers. Source and portable evidence are versioned on `rajeevp/qsa`; the large local ZIP archives remain local.
