<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OLMo QSA on Strix Halo: measured comparison

All paired accuracy phases and four context timing suites are complete.

Original and Changed use the same native hip-ep RTN W4A16 group128 backbone, FP16 backbone/KV and FP32 rotary arithmetic. Only global attention middles 7/11/15/23 change. Layer 19 stays dense. All results are batch one.

The existing step-4096 selector export is reused unchanged in BF16. Training, fine-tuning, recovery, distillation, learned calibration and selector updates: **zero**.

## Accuracy

| Metric | Original | Changed | Change | Paired 95% interval | Gate |
|---|---:|---:|---:|---|---|
| Fresh Dolma PPL | 6.434833 | 6.523952 | +1.385% | [+1.166, +1.632]% | ≤5% increase: Pass point estimate |
| Cached NLL | 1.776616 | 1.794961 | +0.018346 | Not bootstrapped | Diagnostic |
| GSM8K pilot | 94/128 (73.438%) | 91/128 (71.094%) | -2.344 pp | [-6.250, +1.562] pp | Diagnostic; accepted by user |
| GSM token caps | 3 | 3 | — | — | 512-token maximum |

Dolma uses the same 64 document windows × 4096 tokens, 4095 prefill targets and 64 cached targets per document. GSM8K is the reused 128-item, five-shot pilot with greedy decoding, EOS/Question stopping, a strict `####` parser and a 512-token cap. Original/Changed order alternates by document or item. Bootstrap intervals use 10,000 paired resamples, seed 42; they are exploratory, without multiplicity correction. Point-estimate gates do not prove population equivalence. The 4K test does not establish broad long-context quality. These are the same held-out windows reused from sparse-attn, not a new independent port dataset. The user accepted the observed GSM8K result and removed it as an acceptance gate; the original 2-point threshold remains in the archived initial measurement receipt for provenance. Dolma retains its 5% PPL gate.

The final prompt audit found three GSM8K inputs decoded with Windows CP1252 instead of UTF-8. Both variants were rerun for those three inputs; the other 250 rows had identical prompts and were retained. The corrected prompts match the historical source run exactly. The earlier rows and correction receipt are preserved under `excluded-gsm8k-cp1252/`.

## Decode, independent timer

| Prefix tokens | Batch | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |
|---:|---:|---:|---:|---:|---|---:|
| 4,096 | 1 | 26.7399 | 22.2405 | -16.827% | [-17.254, -16.336]% | 6 |
| 16,384 | 1 | 21.9270 | 19.9425 | -9.051% | [-10.233, -8.012]% | 6 |
| 32,768 | 1 | 18.1507 | 17.6363 | -2.834% | [-3.403, -2.270]% | 6 |
| 65,472 | 1 | 12.7704 | 14.1118 | +10.504% | [+9.645, +11.464]% | 6 |

## Prefill, independent timer

| Prefix tokens | Batch | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |
|---:|---:|---:|---:|---:|---|---:|
| 4,096 | 1 | 1714.2162 | 1526.1811 | -10.969% | [-11.572, -10.357]% | 6 |
| 16,384 | 1 | 1494.8061 | 1343.6787 | -10.110% | [-10.769, -9.307]% | 6 |
| 32,768 | 1 | 1404.4801 | 1258.7125 | -10.379% | [-10.592, -10.194]% | 6 |
| 65,472 | 1 | 1211.0101 | 1026.5250 | -15.234% | [-16.003, -14.309]% | 6 |

## Timing-stream continuation diagnostics

| Prefix tokens | Original PPL | Changed PPL | Change |
|---:|---:|---:|---:|
| 4,096 | 3.964028 | 4.054981 | +2.294% |
| 16,384 | 7.758655 | 7.900240 | +1.825% |
| 32,768 | 4.980511 | 6.539904 | +31.310% |
| 65,472 | 6.613280 | 7.668582 | +15.957% |

Each row is one reused timing-stream continuation with 64 targets, computed from untimed warmup logits. These are diagnostics, not independent datasets, retrieval benchmarks or a broad long-context quality guarantee. The 5% acceptance gate applies to the 64-document 4K Dolma test above.

At batch one, aggregate and per-request tokens/s are equal. Rates are total fixed tokens divided by total elapsed time across trials. Positive throughput change means Changed is faster. Prefill and decode use separate HIP-synchronized CPU wall timers. Two exact-shape warmup requests per resident runtime also validate finite, bit-exact replay. Adjacent repeated pairs alternate which variant runs first. Both variants use the same physical GPU, native projection implementation, quantization, token inputs, KV layout/capacity and output-head work.

Decode covers 64 fixed teacher-forced calls, including the output head, and excludes prefill/cache preparation, sampling and tensor/binding creation. Prefill includes the final-position output head and uses a preallocated cache; model load and tensor/binding/cache allocation are excluded. The 64K window uses 65,472 prefix tokens plus 64 decode calls, fitting capacity 65,536. No batched or online-serving performance is claimed. Intervals cover within-run noise, not cross-machine variation.

The combined-runtime 64K attempt reached the 2 GiB available-host-RAM guard after Original's first warmup and was stopped. Its evidence is preserved under `resource-failures/co-resident-65472/` and `stages/paired-64k-performance/`. The 64K fallback uses one resident variant per worker, two warmups per worker, and the same balanced chronological pair order. Equivalent KV buffers are allocated in separate workers; the shorter-context runs share buffers serially. Worker loading and warmups are untimed. Its intervals also include variation between workers. Only completed paired suites enter the timing tables.

## Implementation and numerical checks

The explicit partition manifest leaves all 224 Q/K/V/O/MLP projection nodes byte-identical. Accuracy/operator controls share five backbone sessions. Headline timing preserves the complete native Original graph and includes Changed's boundary/orchestration costs. The timing variants use matched persistent KV layout/capacity and the same native projection implementation. Four dense GQA middles are replaced by a staged wave32 HIP C ABI. Selector projection and parameters remain BF16, with FP32 selector scores/attention accumulation and FP16 native KV/output. Current hidden states are converted from FP16 to BF16 at selector input. No full-history KV gather/cast/copy occurs inside QSA. Full KV remains allocated. Persistent workspaces/cache pointers are checked.

Strict GPU execution (CPU fallback disabled), the same-quantization 4K Transformers reference, complete-versus-partitioned dense equivalence, 200 raw operator/cache cases, deterministic ties and integrated Changed replay/cached checks passed. The correct rotary contract is default RoPE for sliding layers and YaRN for globals, using FP32 tables/arithmetic. Earlier all-YaRN prototypes from Transformers 5.5.3 were excluded and retained as failure evidence. The accepted environment uses 5.19.0.

## Reproduction and evidence

Machine: XSJRAJEEVP04, Windows 11 build 26200, Ryzen AI MAX+ PRO 395, Radeon 8060S gfx1151, wave32, driver 32.0.22050.4002. HIP 7.15.26333, Torch 2.13.0+rocm10.0.0, ORT 1.31.0, Transformers 5.19.0, Python 3.13.13. Cooperative launch is unavailable. The matching baseline build receipt records hip-ep source `e99e1ae046a9dda10417a95135e46e750bc8f3d8`; the task checkout base is `1c9a454141c22bdddfc6071ab8c0c58cb01c36de`. The new staged library's compiler, source hashes and binary digest are recorded separately.

Frozen selector SHA256: `decd3160f4321839c80fee2ba0b0241575fc06017676e38a04286a3ae3daa10b`.

From the repository root, use the task-local interpreter:

```powershell
_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-dolma-quality -- scripts/qsa/evaluate_pair.py dolma
_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-gsm8k-quality -- scripts/qsa/evaluate_pair.py gsm8k
_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-native-4k-performance -- scripts/qsa/benchmark_pair.py --context 4096 --pairs 6
# Repeat the last command with a fresh label and contexts 16384, 32768.
_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-isolated-64k-performance -- scripts/qsa/benchmark_isolated.py --context 65472
_local/qsa/venv/Scripts/python.exe scripts/qsa/report_results.py
```

Stage labels are unique; preserve existing receipts and use new labels for new runs. Completed timing rows are never silently overwritten. Raw artifacts stay under `_local/qsa/`: `paired-quality/`, `paired-performance/`, `stages/` (source snapshots/commands/logs/exits/selector hashes), `environment.json`, `native/build.json`, `qsa-raw-validation.json`, `changed-validation.json` and the accepted `original-w4-rope-v2/` export. The read-only sparse-attn source and its separate remote jobs were not modified or managed by this work. No commit, push or public publication was performed.

See the [port design and baseline audit](../design/olmo-qsa-port.md) for architecture, precision, library hashes and excluded prototype details.
