#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Audit completed paired receipts and render the durable comparison report."""

import json
import math

from prepare_baseline import ROOT, SELECTOR_SHA, WORK


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def interval(values):
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def main():
    quality = WORK / "paired-quality"
    dolma, gsm = (
        read(quality / "dolma-summary.json"),
        read(quality / "gsm8k-summary.json"),
    )
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# OLMo QSA on Strix Halo: measured comparison",
        "",
        (
            "Original and Changed use the same native hip-ep RTN W4A16 group128 backbone, "
            "FP16 backbone/KV and FP32 rotary arithmetic. Only global attention middles "
            "7/11/15/23 change. Layer 19 stays dense. All results are batch one."
        ),
        "",
        (
            "The existing step-4096 selector export is reused unchanged in BF16. "
            "Training, fine-tuning, recovery, distillation, learned calibration and selector updates: **zero**."
        ),
        "",
        "## Accuracy",
        "",
        "| Metric | Original | Changed | Change | Paired 95% interval | Gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    if dolma and dolma["complete"]:
        raw = rows(quality / "dolma.jsonl")
        assert len(raw) == 128
        for variant in ["Original", "Changed"]:
            selected = [r for r in raw if r["variant"] == variant]
            assert {r["document"] for r in selected} == set(range(64)) and len(
                selected
            ) == 64
            assert all(
                r["targets"] == 4095
                and r["cached_targets"] == 64
                and math.isfinite(r["nll_sum"])
                for r in selected
            )
            calculated = math.exp(sum(r["nll_sum"] for r in selected) / (64 * 4095))
            assert abs(calculated - dolma["variants"][variant]["ppl"]) < 1e-10
        a, b = [dolma["variants"][v] for v in ["Original", "Changed"]]
        outcome = "Pass point estimate" if dolma["gate_point_estimate_pass"] else "Fail"
        lines.append(
            f"| Fresh Dolma PPL | {a['ppl']:.6f} | {b['ppl']:.6f} | {dolma['ppl_change_percent']:+.3f}% | {interval(dolma['paired_95_interval_percent'])}% | ≤5% increase: {outcome} |"
        )
        lines.append(
            f"| Cached NLL | {a['cached_nll']:.6f} | {b['cached_nll']:.6f} | {b['cached_nll'] - a['cached_nll']:+.6f} | Not bootstrapped | Diagnostic |"
        )
        integrity = read(quality / "dolma-integrity.json")
        assert (
            integrity["parameters_unchanged"]
            and integrity["export_sha256"] == SELECTOR_SHA
        )
    else:
        lines.append(
            "| Fresh Dolma PPL / cached NLL | Pending paired evaluation | Pending | — | — | Not assessed |"
        )
    if gsm and gsm["complete"]:
        raw = rows(quality / "gsm8k.jsonl")
        assert len(raw) == 256
        for variant in ["Original", "Changed"]:
            selected = [r for r in raw if r["variant"] == variant]
            assert {r["index"] for r in selected} == set(range(128)) and len(
                selected
            ) == 128
            assert (
                sum(r["correct"] for r in selected)
                == gsm["variants"][variant]["correct"]
            )
            assert all(
                len(r["generated_token_ids"]) == r["generated_tokens"] <= 512
                for r in selected
            )
        a, b = [gsm["variants"][v] for v in ["Original", "Changed"]]
        lines.append(
            f"| GSM8K pilot | {a['correct']}/128 ({a['accuracy_percent']:.3f}%) | {b['correct']}/128 ({b['accuracy_percent']:.3f}%) | {gsm['accuracy_change_percentage_points']:+.3f} pp | {interval(gsm['paired_95_interval_percentage_points'])} pp | Diagnostic; accepted by user |"
        )
        lines.append(
            f"| GSM token caps | {a['capped']} | {b['capped']} | — | — | 512-token maximum |"
        )
        integrity = read(quality / "gsm8k-integrity.json")
        assert (
            integrity["parameters_unchanged"]
            and integrity["export_sha256"] == SELECTOR_SHA
        )
    else:
        lines.append("| GSM8K pilot | Pending | Pending | — | — | Not assessed |")
    lines += [
        "",
        (
            "Dolma uses the same 64 document windows × 4096 tokens, 4095 prefill "
            "targets and 64 cached targets per document. GSM8K is the reused 128-item, "
            "five-shot pilot with greedy decoding, EOS/Question stopping, a strict `####` "
            "parser and a 512-token cap. Original/Changed order alternates by document or item. "
            "Bootstrap intervals use 10,000 paired resamples, seed 42; they are exploratory, "
            "without multiplicity correction. Point-estimate gates do not prove population equivalence. "
            "The 4K test does not establish broad long-context quality. These are the same "
            "held-out windows reused from sparse-attn, not a new independent port dataset. "
            "The user accepted the observed GSM8K result and removed it as an acceptance gate; "
            "the original 2-point threshold remains in the archived initial measurement receipt for provenance. "
            "Dolma retains its 5% PPL gate."
        ),
        "",
        (
            "The final prompt audit found three GSM8K inputs decoded with Windows CP1252 "
            "instead of UTF-8. Both variants were rerun for those three inputs; the other "
            "250 rows had identical prompts and were retained. The corrected prompts match "
            "the historical source run exactly. The earlier rows and correction receipt are "
            "preserved under `excluded-gsm8k-cp1252/`."
        ),
        "",
    ]
    summaries = []
    for context in [4096, 16384, 32768, 65472]:
        folder = WORK / "paired-performance" / str(context)
        status, summary = read(folder / "status.json"), read(folder / "summary.json")
        if not status or status["status"] != "complete":
            continue
        raw, protocol = rows(folder / "rows.jsonl"), read(folder / "protocol.json")
        assert protocol.get("original_execution") == "complete native hip-ep graph", (
            "Exclude boundary-only timing diagnostics"
        )
        assert len(raw) == 2 * protocol["pairs"]
        assert status["selector_integrity"]["parameters_unchanged"]
        assert status["cache_pointers_unchanged"]
        assert status[
            "isolated_workers_sessions_unchanged"
            if protocol.get("process_isolation")
            else "shared_sessions_unchanged"
        ]
        for variant in ["Original", "Changed"]:
            chosen = [r for r in raw if r["variant"] == variant]
            assert {r["pair"] for r in chosen} == set(range(protocol["pairs"]))
            assert sum(r["order"] == 0 for r in chosen) == protocol["pairs"] // 2
            for phase in ["prefill", "decode"]:
                mean = sum(r[phase + "_seconds"] for r in chosen) / len(chosen)
                assert (
                    abs(mean - summary["phases"][phase][variant]["mean_seconds"]) < 1e-9
                )
        summaries.append(summary)
    for phase in ["decode", "prefill"]:
        lines += [
            f"## {phase.capitalize()}, independent timer",
            "",
            "| Prefix tokens | Batch | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |",
            "|---:|---:|---:|---:|---:|---|---:|",
        ]
        for summary in summaries:
            record = summary["phases"][phase]
            lines.append(
                f"| {summary['context']:,} | 1 | {record['Original']['aggregate_tokens_s']:.4f} | {record['Changed']['aggregate_tokens_s']:.4f} | {record['throughput_change_percent']:+.3f}% | {interval(record['paired_95_interval_percent'])}% | {summary['pairs']} |"
            )
        if not summaries:
            lines.append("| Pending paired timings | 1 | — | — | — | — | — |")
        lines.append("")
    finished = bool(
        dolma and dolma["complete"] and gsm and gsm["complete"] and len(summaries) == 4
    )
    lines[6:6] = [
        (
            "All paired accuracy phases and four context timing suites are complete."
            if finished
            else "Evaluation is in progress. Tables include completed paired phases only."
        ),
        "",
    ]
    lines += [
        "## Timing-stream continuation diagnostics",
        "",
        "| Prefix tokens | Original PPL | Changed PPL | Change |",
        "|---:|---:|---:|---:|",
    ]
    continuation_count = 0
    for summary in summaries:
        validation = read(
            WORK
            / "paired-performance"
            / str(summary["context"])
            / "model-validation.json"
        )
        if all("continuation_ppl" in validation[v] for v in ["Original", "Changed"]):
            a, b = [validation[v]["continuation_ppl"] for v in ["Original", "Changed"]]
            lines.append(
                f"| {summary['context']:,} | {a:.6f} | {b:.6f} | {100 * (b / a - 1):+.3f}% |"
            )
            continuation_count += 1
    if not continuation_count:
        lines.append("| Pending | — | — | — |")
    lines += [
        "",
        (
            "Each row is one reused timing-stream continuation with 64 targets, "
            "computed from untimed warmup logits. These are diagnostics, not independent "
            "datasets, retrieval benchmarks or a broad long-context quality guarantee. "
            "The 5% acceptance gate applies to the 64-document 4K Dolma test above."
        ),
        "",
    ]
    lines += [
        (
            "At batch one, aggregate and per-request tokens/s are equal. Rates are total "
            "fixed tokens divided by total elapsed time across trials. Positive throughput "
            "change means Changed is faster. Prefill and decode use separate HIP-synchronized "
            "CPU wall timers. Two exact-shape warmup requests per resident runtime also validate finite, "
            "bit-exact replay. Adjacent repeated pairs alternate which variant runs first. "
            "Both variants use the same physical GPU, native projection implementation, quantization, "
            "token inputs, KV layout/capacity and output-head work."
        ),
        "",
        (
            "Decode covers 64 fixed teacher-forced calls, including the output head, and excludes "
            "prefill/cache preparation, sampling and tensor/binding creation. Prefill includes "
            "the final-position output head and uses a preallocated cache; model load and "
            "tensor/binding/cache allocation are excluded. The 64K window uses 65,472 prefix "
            "tokens plus 64 decode calls, fitting capacity 65,536. No batched or online-serving "
            "performance is claimed. Intervals cover within-run noise, not cross-machine variation."
        ),
        "",
        (
            "The combined-runtime 64K attempt reached the 2 GiB available-host-RAM guard "
            "after Original's first warmup and was stopped. Its evidence is preserved under "
            "`resource-failures/co-resident-65472/` and `stages/paired-64k-performance/`. "
            "The 64K fallback uses one resident variant per worker, two warmups per worker, "
            "and the same balanced chronological pair order. Equivalent KV buffers are allocated "
            "in separate workers; the shorter-context runs share buffers serially. Worker loading "
            "and warmups are untimed. Its intervals also include variation between workers. "
            "Only completed paired suites enter the timing tables."
        ),
        "",
        "## Implementation and numerical checks",
        "",
        (
            "The explicit partition manifest leaves all 224 Q/K/V/O/MLP projection nodes "
            "byte-identical. Accuracy/operator controls share five backbone sessions. "
            "Headline timing preserves the complete native Original graph and includes "
            "Changed's boundary/orchestration costs. The timing variants use matched "
            "persistent KV layout/capacity and the same native projection implementation. Four dense GQA "
            "middles are replaced by a staged wave32 HIP C ABI. Selector projection and "
            "parameters remain BF16, with FP32 selector scores/attention accumulation and "
            "FP16 native KV/output. Current hidden states are converted from FP16 to BF16 "
            "at selector input. No full-history KV gather/cast/copy occurs inside QSA. "
            "Full KV remains allocated. Persistent workspaces/cache pointers are checked."
        ),
        "",
        (
            "Strict GPU execution (CPU fallback disabled), the same-quantization 4K Transformers "
            "reference, complete-versus-partitioned dense equivalence, 200 raw operator/cache "
            "cases, deterministic ties and integrated Changed replay/cached checks passed. "
            "The correct rotary contract is default RoPE for sliding layers and YaRN for globals, "
            "using FP32 tables/arithmetic. Earlier all-YaRN prototypes from Transformers 5.5.3 "
            "were excluded and retained as failure evidence. The accepted environment uses 5.19.0."
        ),
        "",
        "## Reproduction and evidence",
        "",
        (
            "Machine: XSJRAJEEVP04, Windows 11 build 26200, Ryzen AI MAX+ PRO 395, Radeon 8060S "
            "gfx1151, wave32, driver 32.0.22050.4002. HIP 7.15.26333, Torch 2.13.0+rocm10.0.0, "
            "ORT 1.31.0, Transformers 5.19.0, Python 3.13.13. Cooperative launch is unavailable. "
            "The matching baseline build receipt records hip-ep source "
            "`e99e1ae046a9dda10417a95135e46e750bc8f3d8`; the task checkout base is "
            "`1c9a454141c22bdddfc6071ab8c0c58cb01c36de`. The new staged library's compiler, "
            "source hashes and binary digest are recorded separately."
        ),
        "",
        f"Frozen selector SHA256: `{SELECTOR_SHA}`.",
        "",
        "From the repository root, use the task-local interpreter:",
        "",
        "```powershell",
        "_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-dolma-quality -- scripts/qsa/evaluate_pair.py dolma",
        "_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-gsm8k-quality -- scripts/qsa/evaluate_pair.py gsm8k",
        "_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-native-4k-performance -- scripts/qsa/benchmark_pair.py --context 4096 --pairs 6",
        "# Repeat the last command with a fresh label and contexts 16384, 32768.",
        "_local/qsa/venv/Scripts/python.exe scripts/qsa/run_stage.py paired-isolated-64k-performance -- scripts/qsa/benchmark_isolated.py --context 65472",
        "_local/qsa/venv/Scripts/python.exe scripts/qsa/report_results.py",
        "```",
        "",
        (
            "Stage labels are unique; preserve existing receipts and use new labels for "
            "new runs. Completed timing rows are never silently overwritten. Raw artifacts "
            "stay under `_local/qsa/`: `paired-quality/`, `paired-performance/`, "
            "`stages/` (source snapshots/commands/logs/exits/selector hashes), "
            "`environment.json`, `native/build.json`, `qsa-raw-validation.json`, "
            "`changed-validation.json` and the accepted `original-w4-rope-v2/` export. "
            "The read-only sparse-attn source and its separate remote jobs were not modified "
            "or managed by this work. No commit, push or public publication was performed."
        ),
        "",
        (
            "See the [port design and baseline audit](../design/olmo-qsa-port.md) for "
            "architecture, precision, library hashes and excluded prototype details."
        ),
        "",
    ]
    output = ROOT / "docs/experiments/olmo-qsa-results.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
