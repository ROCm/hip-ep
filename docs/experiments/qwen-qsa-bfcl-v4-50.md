<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen native BFCL v4: paired 50-case pilot

The same pinned Qwen/Qwen3.8-27B W4A16 backbone runs with default attention and frozen QSA at layers27/31/35/39 on Radeon8060S/gfx1151. Batch1, one resident full-model session at a time, measured native binaries, no training or calibration. All50 cases run in both variants.

The seed42 selection contains two cases from each of13 single-turn categories and six from each of four multi-turn categories:26 single-turn and24 multi-turn. The public AST checker control is excluded. This is an unweighted component pilot, **not the official BFCLv4 overall score**. Web-search, memory and format-sensitivity components are excluded. The [official BFCL source](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard) supplies prompts, parsers, simulator backends and scoring.

| Default correct | QSA correct | QSA minus default | Paired stratified95% interval |
|---:|---:|---:|---|
| 26/50 (52.0%) | 24/50 (48.0%) | -4.0pp | [-12.0, +4.0]pp |

Outcomes differ in6/50 pairs. The interval uses10,000 paired resamples within category, seed42. This small diagnostic is nongating and does not establish population equivalence. Infrastructure failures are retained separately and never scored as wrong answers.

## Per-component results

| Component | Cases | Default correct | QSA correct |
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
| `multi_turn_miss_func` | 6 | 1 | 0 |
| `multi_turn_miss_param` | 6 | 1 | 0 |
| `multi_turn_long_context` | 6 | 1 | 3 |

## Actual workload lengths and turns

Prompts include the actual Qwen chat template, tool/function descriptions, conversation history and tool results. Thinking is disabled. Generation is greedy with both official EOS IDs and a1,024-token cap per model response, within a32,768-token cache. No constrained decoding or benchmark answers are supplied. Initial prompts are identical between variants; later histories can diverge.

A turn is a benchmark user-interaction round; a generation call is one assistant response, so a multi-step tool interaction can make several calls within a turn. Token sums include repeatedly processed history; they are not unique context lengths. Completion tokens include an emitted EOS. Per-case and per-generation CSVs retain exact counts, cap flags and pruning eligibility.

| Metric | Default | QSA |
|---|---:|---:|
| Declared turns/case (min / median / p95 / max) | 1 / 1 / 6.0 / 7 | 1 / 1 / 6.0 / 7 |
| Observed turns/case (min / median / p95 / max) | 1 / 1 / 6.0 / 7 | 1 / 1 / 6.0 / 6 |
| Generation calls/case (min / median / p95 / max) | 1 / 1 / 15.5 / 33 | 1 / 1 / 20.5 / 22 |
| Prompt tokens/generation (min / median / p95 / max) | 431 / 7417.5 / 11977.0 / 17407 | 431 / 7333 / 15039.0 / 15795 |
| Response tokens/generation (min / median / p95 / max) | 4 / 23 / 163.7 / 749 | 4 / 25 / 124.8 / 738 |
| Total response tokens/case (min / median / p95 / max) | 11 / 99.5 / 952.4 / 2028 | 11 / 115.5 / 936.6 / 2741 |
| Prompt+response tokens/generation (min / median / p95 / max) | 475 / 7461 / 11989.5 / 17450 | 475 / 7352 / 15071.2 / 15848 |
| Generation calls | 294 | 283 |
| Capped generations | 0 | 0 |
| Cases with any capped response | 0 | 0 |
| Responses with reduced context budget | 0 | 0 |
| Cases reaching the pruning threshold | 24 | 24 |

QSA actually prunes in24/50 cases and257/283 generation calls. Pruning first occurs at query position2052 (zero-based); requests that never reach it test full-support behavior. Inspect quality differences against this coverage and the cap counts before drawing sparse-attention conclusions.

Accuracy trajectories use each variant's own outputs and tool paths. Their elapsed times and token totals are not a fixed-work throughput comparison. Default runs first and QSA second to keep one model resident; no latency speedup is inferred.

<!-- BFCL_WORKLOAD_ASSESSMENT_BEGIN -->
## Pruning groups and stopping limits

These groups use the frozen initial prompt lengths, selected before generation. Actual later-history pruning coverage is reported above.

| Initial prompt group | Cases | Default correct | QSA correct |
|---|---:|---:|---:|
| Prefill reaches sparse pruning | 24 | 6 | 4 |
| Prefill stays within full support | 26 | 20 | 20 |

BFCL's official `MAXIMUM_STEP_LIMIT=20` setting and pinned stopping semantics were retained. Its step bound stops **2 default cases and3 QSA cases**, separately from the response-token caps above. A step-limit stop can end a case before all declared turns are reached; the official checker still scores the resulting trajectory. No stopped cases are replaced.

`turns.csv` records each observed benchmark turn, its model-call count, prompt-token totals/range, response-token totals/maximum and step-limit flag. The independent token audit verifies the per-turn nesting against every actual generation's counts. `token-audit.json` retains the stopped case IDs and subgroup numerators/denominators.
<!-- BFCL_WORKLOAD_ASSESSMENT_END -->

## Evidence and reproduction

Raw selection, checker controls, protocol, source snapshots, per-case results, complete request messages/rendered prompts/token IDs, worker/client logs and independently recomputed official scores are under `_local/qwen-qsa/bfcl-v4-50/`. `summary.json`, `cases.csv`, `generations.csv` and `categories.csv` retain the report data. Prior OLMo BFCL results are unchanged.

`scripts/qwen_qsa/run_bfcl.py` supervises serial GPU work and the isolated WSL CPU scorer, with a4GiB available-RAM guard, four-hour stage bound, atomic queue files and supervisor leases. Completed case rows are checkpointed; partial raw generations remain retained. `bfcl_client.py --audit` independently rescores all100 variant rows. `report_bfcl.py` generates this report from the audited records. See the [main paper handoff](../reproducibility/qwen-hip-ep/README.md) for binary/source provenance and the immutable raw archive.
