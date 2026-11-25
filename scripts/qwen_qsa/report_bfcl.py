#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Report audited BFCL quality together with actual token, turn and pruning coverage."""

import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "_local/qwen-qsa/bfcl-v4-50"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def csv_file(path, rows):
    assert rows
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        position - low
    )


def distribution(values):
    assert values
    return dict(
        count=len(values),
        minimum=min(values),
        median=statistics.median(values),
        mean=statistics.mean(values),
        p95=percentile(values, 0.95),
        maximum=max(values),
        total=sum(values),
    )


def main():
    audit = load(TASK / "audit.json")
    assert audit["status"] == "passed" and audit["cases"] == 50
    assert audit["rows_sha256"] == sha(TASK / "rows.jsonl")
    assert audit["protocol_sha256"] == sha(TASK / "protocol.json")
    rows = [json.loads(s) for s in (TASK / "rows.jsonl").read_text().splitlines()]
    generations, cases = [], []
    for row in rows:
        responses = [
            load(ROOT / row["queue"] / (i + ".response.json"))
            for i in row["request_ids"]
        ]
        for response in responses:
            assert response["case_id"] == row["id"]
            generations.append(
                {
                    key: response[key]
                    for key in [
                        "variant",
                        "case_id",
                        "category",
                        "generation_index",
                        "request_id",
                        "prompt_tokens",
                        "prompt_characters",
                        "completion_tokens",
                        "input_plus_output_tokens",
                        "finish_reason",
                        "maximum_new_tokens",
                        "context_budget_reduced",
                        "generation_seconds",
                        "prefill_pruning_eligible",
                        "pruning_eligible",
                        "qsa_pruning_exercised",
                    ]
                }
                | {
                    "response_characters": len(response["text"]),
                    **{
                        role + "_messages": count
                        for role, count in response["message_role_counts"].items()
                    },
                }
            )
        cases.append(
            dict(
                variant=row["variant"],
                id=row["id"],
                category=row["category"],
                correct=row["correct"],
                declared_turns=row["declared_turns"],
                declared_user_messages=row["declared_user_messages"],
                function_count=row["function_count"],
                observed_turns=len(row["result"])
                if row["category"].startswith("multi_turn_")
                else 1,
                generation_calls=len(responses),
                initial_prompt_tokens=responses[0]["prompt_tokens"],
                minimum_prompt_tokens=min(r["prompt_tokens"] for r in responses),
                maximum_prompt_tokens=max(r["prompt_tokens"] for r in responses),
                total_prompt_tokens=sum(r["prompt_tokens"] for r in responses),
                total_completion_tokens=sum(r["completion_tokens"] for r in responses),
                maximum_completion_tokens=max(
                    r["completion_tokens"] for r in responses
                ),
                maximum_input_plus_output_tokens=max(
                    r["input_plus_output_tokens"] for r in responses
                ),
                capped_generations=sum(
                    r["finish_reason"] == "length" for r in responses
                ),
                reduced_context_budget_generations=sum(
                    r["context_budget_reduced"] for r in responses
                ),
                pruning_eligible=any(r["pruning_eligible"] for r in responses),
                qsa_pruning_exercised=any(
                    r["qsa_pruning_exercised"] for r in responses
                ),
            )
        )
    assert len(cases) == 100
    indexed = {(r["variant"], r["id"]): r for r in cases}
    categories = list(dict.fromkeys(r["category"] for r in cases))
    category_rows, deltas = [], defaultdict(list)
    for category in categories:
        selected = [
            r for r in cases if r["category"] == category and r["variant"] == "Original"
        ]
        ac = sum(r["correct"] for r in selected)
        bc = sum(indexed["Changed", r["id"]]["correct"] for r in selected)
        category_rows.append(
            dict(
                category=category,
                cases=len(selected),
                default_correct=ac,
                qsa_correct=bc,
            )
        )
        deltas[category] = [
            int(indexed["Changed", r["id"]]["correct"]) - int(r["correct"])
            for r in selected
        ]
    rng = random.Random(42)
    changes = []
    for _ in range(10000):
        changes.append(
            100
            / 50
            * sum(rng.choice(values) for values in deltas.values() for _ in values)
        )
    variants = {}
    for variant in ["Original", "Changed"]:
        c = [r for r in cases if r["variant"] == variant]
        g = [r for r in generations if r["variant"] == variant]
        variants[variant] = dict(
            correct=sum(r["correct"] for r in c),
            cases=50,
            accuracy_percent=2 * sum(r["correct"] for r in c),
            generation_calls=len(g),
            cases_with_caps=sum(r["capped_generations"] > 0 for r in c),
            capped_generations=sum(r["finish_reason"] == "length" for r in g),
            context_budget_reduced_generations=sum(
                r["context_budget_reduced"] for r in g
            ),
            pruning_eligible_cases=sum(r["pruning_eligible"] for r in c),
            pruning_eligible_generations=sum(r["pruning_eligible"] for r in g),
            actual_qsa_pruning_cases=sum(r["qsa_pruning_exercised"] for r in c),
            declared_turns=distribution([r["declared_turns"] for r in c]),
            observed_turns=distribution([r["observed_turns"] for r in c]),
            generation_calls_per_case=distribution([r["generation_calls"] for r in c]),
            prompt_tokens_per_generation=distribution([r["prompt_tokens"] for r in g]),
            completion_tokens_per_generation=distribution(
                [r["completion_tokens"] for r in g]
            ),
            completion_tokens_per_case=distribution(
                [r["total_completion_tokens"] for r in c]
            ),
            input_plus_output_tokens_per_generation=distribution(
                [r["input_plus_output_tokens"] for r in g]
            ),
        )
    changed_pairs = sum(
        indexed["Original", r["id"]]["correct"] != r["correct"]
        for r in cases
        if r["variant"] == "Changed"
    )
    summary = dict(
        status="complete",
        benchmark="BFCL v4 stratified 50-case component pilot",
        official_overall_score=False,
        acceptance_gate_enabled=False,
        cases=50,
        variant_rows=100,
        variants=variants,
        categories=category_rows,
        accuracy_change_percentage_points=variants["Changed"]["accuracy_percent"]
        - variants["Original"]["accuracy_percent"],
        paired_stratified_bootstrap_95_interval_percentage_points=[
            percentile(changes, 0.025),
            percentile(changes, 0.975),
        ],
        bootstrap_resamples=10000,
        bootstrap_seed=42,
        outcome_disagreement_pairs=changed_pairs,
        initial_prompt_tokens=distribution(
            [r["initial_prompt_tokens"] for r in cases if r["variant"] == "Original"]
        ),
        declared_turn_histogram=dict(
            Counter(r["declared_turns"] for r in cases if r["variant"] == "Original")
        ),
        function_count=distribution(
            [r["function_count"] for r in cases if r["variant"] == "Original"]
        ),
        protocol_sha256=sha(TASK / "protocol.json"),
        manifest_sha256=sha(TASK / "manifest.json"),
        rows_sha256=sha(TASK / "rows.jsonl"),
        audit_sha256=sha(TASK / "audit.json"),
        training_updates=0,
    )
    write(TASK / "summary.json", summary)
    csv_file(TASK / "cases.csv", cases)
    csv_file(TASK / "generations.csv", generations)
    csv_file(TASK / "categories.csv", category_rows)
    a, b = variants["Original"], variants["Changed"]
    ci = summary["paired_stratified_bootstrap_95_interval_percentage_points"]
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Qwen native BFCL v4: paired 50-case pilot",
        "",
        "The same pinned Qwen/Qwen3.8-27B W4A16 backbone runs with default attention and frozen QSA at layers27/31/35/39 on Radeon8060S/gfx1151. Batch1, one resident full-model session at a time, measured native binaries, no training or calibration. All50 cases run in both variants.",
        "",
        "The seed42 selection contains two cases from each of13 single-turn categories and six from each of four multi-turn categories:26 single-turn and24 multi-turn. The public AST checker control is excluded. This is an unweighted component pilot, **not the official BFCLv4 overall score**. Web-search, memory and format-sensitivity components are excluded. The [official BFCL source](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard) supplies prompts, parsers, simulator backends and scoring.",
        "",
        "| Default correct | QSA correct | QSA minus default | Paired stratified95% interval |",
        "|---:|---:|---:|---|",
        f"| {a['correct']}/50 ({a['accuracy_percent']:.1f}%) | {b['correct']}/50 ({b['accuracy_percent']:.1f}%) | {summary['accuracy_change_percentage_points']:+.1f}pp | [{ci[0]:+.1f}, {ci[1]:+.1f}]pp |",
        "",
        f"Outcomes differ in{changed_pairs}/50 pairs. The interval uses10,000 paired resamples within category, seed42. This small diagnostic is nongating and does not establish population equivalence. Infrastructure failures are retained separately and never scored as wrong answers.",
        "",
        "## Per-component results",
        "",
        "| Component | Cases | Default correct | QSA correct |",
        "|---|---:|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| `{row['category']}` | {row['cases']} | {row['default_correct']} | {row['qsa_correct']} |"
        )
    lines += [
        "",
        "## Actual workload lengths and turns",
        "",
        "Prompts include the actual Qwen chat template, tool/function descriptions, conversation history and tool results. Thinking is disabled. Generation is greedy with both official EOS IDs and a1,024-token cap per model response, within a32,768-token cache. No constrained decoding or benchmark answers are supplied. Initial prompts are identical between variants; later histories can diverge.",
        "",
        "A turn is a benchmark user-interaction round; a generation call is one assistant response, so a multi-step tool interaction can make several calls within a turn. Token sums include repeatedly processed history; they are not unique context lengths. Completion tokens include an emitted EOS. Per-case and per-generation CSVs retain exact counts, cap flags and pruning eligibility.",
        "",
        "| Metric | Default | QSA |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Declared turns/case", "declared_turns"),
        ("Observed turns/case", "observed_turns"),
        ("Generation calls/case", "generation_calls_per_case"),
        ("Prompt tokens/generation", "prompt_tokens_per_generation"),
        ("Response tokens/generation", "completion_tokens_per_generation"),
        ("Total response tokens/case", "completion_tokens_per_case"),
        (
            "Prompt+response tokens/generation",
            "input_plus_output_tokens_per_generation",
        ),
    ]:

        def fmt(v):
            d = variants[v][key]
            return f"{d['minimum']} / {d['median']:g} / {d['p95']:.1f} / {d['maximum']}"

        lines.append(
            f"| {label} (min / median / p95 / max) | {fmt('Original')} | {fmt('Changed')} |"
        )
    for label, key in [
        ("Generation calls", "generation_calls"),
        ("Capped generations", "capped_generations"),
        ("Cases with any capped response", "cases_with_caps"),
        ("Responses with reduced context budget", "context_budget_reduced_generations"),
        ("Cases reaching the pruning threshold", "pruning_eligible_cases"),
    ]:
        lines.append(f"| {label} | {a[key]} | {b[key]} |")
    lines += [
        "",
        f"QSA actually prunes in{b['actual_qsa_pruning_cases']}/50 cases and{b['pruning_eligible_generations']}/{b['generation_calls']} generation calls. Pruning first occurs at query position2052 (zero-based); requests that never reach it test full-support behavior. Inspect quality differences against this coverage and the cap counts before drawing sparse-attention conclusions.",
        "",
        "Accuracy trajectories use each variant's own outputs and tool paths. Their elapsed times and token totals are not a fixed-work throughput comparison. Default runs first and QSA second to keep one model resident; no latency speedup is inferred.",
        "",
        "## Evidence and reproduction",
        "",
        "Raw selection, checker controls, protocol, source snapshots, per-case results, complete request messages/rendered prompts/token IDs, worker/client logs and independently recomputed official scores are under `_local/qwen-qsa/bfcl-v4-50/`. `summary.json`, `cases.csv`, `generations.csv` and `categories.csv` retain the report data. Prior OLMo BFCL results are unchanged.",
        "",
        "`scripts/qwen_qsa/run_bfcl.py` supervises serial GPU work and the isolated WSL CPU scorer, with a4GiB available-RAM guard, four-hour stage bound, atomic queue files and supervisor leases. Completed case rows are checkpointed; partial raw generations remain retained. `bfcl_client.py --audit` independently rescores all100 variant rows. `report_bfcl.py` generates this report from the audited records. See the [main paper handoff](../reproducibility/qwen-hip-ep/README.md) for binary/source provenance and the immutable raw archive.",
        "",
    ]
    path = ROOT / "docs/experiments/qwen-qsa-bfcl-v4-50.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    from performance_docs import update_documents

    update_documents()
    print(
        json.dumps(
            {
                v: dict(
                    correct=d["correct"],
                    generation_calls=d["generation_calls"],
                    caps=d["capped_generations"],
                )
                for v, d in variants.items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
