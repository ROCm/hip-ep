#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Audit/export Eight BFCL50 against unchanged reused Original/four records."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile
from datetime import datetime, timezone

from audit_bfcl_tokens import step_limit_logs
from report_bfcl import distribution, percentile
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "bfcl-eight-v4-50"
BASELINE = WORK / "bfcl-v4-50"
PACKET = ROOT / "docs/reproducibility/qwen-eight-hip-ep"
REPORT = ROOT / "docs/experiments/qwen-qsa-eight-bfcl-v4-50.md"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_file(p, data):
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def compare(records, candidate):
    indexed = {(r["variant"], r["id"]): r for r in records}
    original = [r for r in records if r["variant"] == "Original"]
    assert len(original) == 50
    groups = {}
    categories = []
    for category in dict.fromkeys(r["category"] for r in original):
        selected = [r for r in original if r["category"] == category]
        differences = [
            int(indexed[candidate, r["id"]]["correct"]) - int(r["correct"])
            for r in selected
        ]
        groups[category] = differences
        categories.append(
            dict(
                category=category,
                cases=len(selected),
                original_correct=sum(r["correct"] for r in selected),
                candidate_correct=sum(
                    indexed[candidate, r["id"]]["correct"] for r in selected
                ),
            )
        )
    rng = random.Random(42)
    changes = sorted(
        2 * sum(rng.choice(v) for v in groups.values() for _ in v) for _ in range(10000)
    )
    totals = {
        v: sum(r["correct"] for r in records if r["variant"] == v)
        for v in ["Original", candidate]
    }
    return dict(
        correct=totals,
        delta_percentage_points=2 * (totals[candidate] - totals["Original"]),
        paired_stratified_95_interval_percentage_points=[
            percentile(changes, 0.025),
            percentile(changes, 0.975),
        ],
        categories=categories,
        disagreement_pairs=sum(
            indexed[candidate, r["id"]]["correct"] != r["correct"] for r in original
        ),
        bootstrap_resamples=10000,
        bootstrap_seed=42,
    )


def collect(records):
    cases, generations = [], []
    for row in records:
        responses = [
            load(ROOT / row["queue"] / (i + ".response.json"))
            for i in row["request_ids"]
        ]
        assert responses and all(
            r["case_id"] == row["id"] and r["variant"] == row["variant"]
            for r in responses
        )
        for r in responses:
            generations.append(
                {
                    k: r[k]
                    for k in [
                        "variant",
                        "case_id",
                        "category",
                        "generation_index",
                        "request_id",
                        "prompt_tokens",
                        "completion_tokens",
                        "input_plus_output_tokens",
                        "finish_reason",
                        "maximum_new_tokens",
                        "context_budget_reduced",
                        "generation_seconds",
                        "prefill_pruning_eligible",
                        "pruning_eligible",
                        "qsa_pruning_exercised",
                        "finite",
                        "cache_pointers_unchanged",
                    ]
                }
            )
        cases.append(
            dict(
                variant=row["variant"],
                id=row["id"],
                category=row["category"],
                correct=row["correct"],
                declared_turns=row["declared_turns"],
                observed_turns=len(row["result"])
                if row["category"].startswith("multi_turn_")
                else 1,
                generation_calls=len(responses),
                initial_prompt_tokens=responses[0]["prompt_tokens"],
                total_prompt_tokens=sum(r["prompt_tokens"] for r in responses),
                maximum_prompt_tokens=max(r["prompt_tokens"] for r in responses),
                total_completion_tokens=sum(r["completion_tokens"] for r in responses),
                maximum_completion_tokens=max(
                    r["completion_tokens"] for r in responses
                ),
                capped_generations=sum(
                    r["finish_reason"] == "length" for r in responses
                ),
                context_budget_reductions=sum(
                    r["context_budget_reduced"] for r in responses
                ),
                official_step_limit_hit=bool(list(step_limit_logs(row["metadata"]))),
                actual_qsa_pruning=any(r["qsa_pruning_exercised"] for r in responses),
                initially_pruned=responses[0]["prompt_tokens"] > 2052,
            )
        )
    variants = {}
    for variant in dict.fromkeys(r["variant"] for r in cases):
        c = [r for r in cases if r["variant"] == variant]
        g = [r for r in generations if r["variant"] == variant]
        variants[variant] = dict(
            cases=len(c),
            correct=sum(r["correct"] for r in c),
            accuracy_percent=100 * sum(r["correct"] for r in c) / len(c),
            generation_calls=len(g),
            prompt_tokens=distribution([r["prompt_tokens"] for r in g]),
            completion_tokens=distribution([r["completion_tokens"] for r in g]),
            declared_turns=distribution([r["declared_turns"] for r in c]),
            observed_turns=distribution([r["observed_turns"] for r in c]),
            calls_per_case=distribution([r["generation_calls"] for r in c]),
            capped_generations=sum(r["capped_generations"] for r in c),
            context_budget_reductions=sum(r["context_budget_reductions"] for r in c),
            step_limited_cases=sum(r["official_step_limit_hit"] for r in c),
            actual_pruned_cases=sum(r["actual_qsa_pruning"] for r in c),
            actual_pruned_generations=sum(r["qsa_pruning_exercised"] for r in g),
            initial_groups={
                label: dict(
                    cases=sum(r["initially_pruned"] == flag for r in c),
                    correct=sum(
                        r["correct"] for r in c if r["initially_pruned"] == flag
                    ),
                )
                for label, flag in [("pruned", True), ("full_support", False)]
            },
        )
    return cases, generations, variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        old = rows(BASELINE / "rows.jsonl")
        result = compare(old, "Changed")
        cases, generations, variants = collect(old)
        assert result["correct"] == {"Original": 26, "Changed": 24}
        assert result["paired_stratified_95_interval_percentage_points"] == [-12.0, 4.0]
        assert len(cases) == 100 and len(generations) == 577
        assert (
            variants["Original"]["generation_calls"] == 294
            and variants["Changed"]["generation_calls"] == 283
        )
        save(
            TASK / "reporter-self-test.json",
            dict(
                status="passed",
                scope="Historical four-selector records only; no new score asserted",
                source_rows_sha256=sha(BASELINE / "rows.jsonl"),
                case_rows=100,
                generation_rows=577,
                known_bootstrap_interval_reproduced=True,
            ),
        )
        print(
            "Historical arithmetic/bootstrap/workload self-test passed; no GPU initialized"
        )
        return
    assert not (ROOT / "_local/qsa/gpu-stage.lock").exists()
    protocol = load(TASK / "protocol.json")
    audit = load(TASK / "audit.json")
    token = load(TASK / "token-audit.json")
    assert (
        audit["status"] == token["status"] == "passed"
        and audit["rows"] == 50
        and token["complete"]
    )
    assert audit["rows_sha256"] == token["rows_sha256"] == sha(TASK / "rows.jsonl")
    assert (
        audit["protocol_sha256"]
        == token["protocol_sha256"]
        == sha(TASK / "protocol.json")
    )
    assert protocol["reused_original_rows_sha256"] == sha(
        TASK / "original-reused.jsonl"
    )
    original = rows(TASK / "original-reused.jsonl")
    eight = rows(TASK / "rows.jsonl")
    assert len(original) == len(eight) == 50
    assert {r["id"] for r in original} == {r["id"] for r in eight}
    assert all(r["variant"] == "Eight" for r in eight)
    data = original + eight
    comparison = compare(data, "Eight")
    cases, generations, variants = collect(data)
    old = load(BASELINE / "summary.json")
    summary = dict(
        state="complete",
        benchmark="BFCL v4 fixed50-case component pilot",
        official_overall_score=False,
        acceptance_gate_enabled=False,
        original_reused=True,
        four_reused_correct=old["variants"]["Changed"]["correct"],
        variants=variants,
        comparison=comparison,
        selector_layers=protocol["selector_layers"],
        selector_export_sha256=protocol["selector_export_sha256"],
        manifest_sha256=sha(TASK / "manifest.json"),
        protocol_sha256=sha(TASK / "protocol.json"),
        new_rows_sha256=sha(TASK / "rows.jsonl"),
        reused_original_rows_sha256=sha(TASK / "original-reused.jsonl"),
        official_audit_sha256=sha(TASK / "audit.json"),
        token_audit_sha256=sha(TASK / "token-audit.json"),
        generation_elapsed_seconds=load(TASK / "status.json")["elapsed_seconds"],
        raw_data_policy="Original row bytes are unchanged; each reused/new row retains its own protocol and queue/source references",
    )
    save(TASK / "summary.json", summary)
    save(PACKET / "bfcl-eight-summary.json", summary)
    csv_file(TASK / "cases.csv", cases)
    csv_file(TASK / "generations.csv", generations)
    csv_file(TASK / "categories.csv", comparison["categories"])
    out = PACKET / "raw/bfcl-eight-v4-50"
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.json",
        "protocol.json",
        "manifest.json",
        "audit.json",
        "token-audit.json",
        "reporter-self-test.json",
        "cases.csv",
        "generations.csv",
        "categories.csv",
        "turns.csv",
        "initial-profile.json",
        "checker-self-test.json",
    ]:
        shutil.copyfile(TASK / name, out / name)
    raw = (TASK / "original-reused.jsonl").read_bytes() + (
        TASK / "rows.jsonl"
    ).read_bytes()
    (TASK / "combined-rows.jsonl").write_bytes(raw)
    parts = out / "row-parts"
    parts.mkdir(exist_ok=True)
    index = []
    for n, start in enumerate(range(0, len(raw), 256 * 1024)):
        chunk = raw[start : start + 256 * 1024]
        path = parts / f"{n:05d}.bin.gz"
        path.write_bytes(gzip.compress(chunk, mtime=0))
        assert path.stat().st_size < 500 * 1024
        index.append(
            dict(
                path=path.relative_to(out).as_posix(),
                sha256=sha(path),
                uncompressed_sha256=hashlib.sha256(chunk).hexdigest(),
                bytes=len(chunk),
            )
        )
    assert (
        b"".join(gzip.decompress((out / r["path"]).read_bytes()) for r in index) == raw
    )
    save(
        out / "rows-parts.json",
        dict(
            rows=100,
            original_reused_rows=50,
            new_eight_rows=50,
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
            reconstruction="Decompress each part,then concatenate raw bytes in listed order before decoding UTF-8/JSONL",
            parts=index,
        ),
    )
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Qwen eight-selector BFCL v4: fixed 50-case pilot",
        "",
        f"**Eight: {variants['Eight']['correct']}/50 ({variants['Eight']['accuracy_percent']:.1f}%)**, versus reused Original **{variants['Original']['correct']}/50 ({variants['Original']['accuracy_percent']:.1f}%)** and historical four-selector **{summary['four_reused_correct']}/50**. Delta versus Original is **{comparison['delta_percentage_points']:+.1f} percentage points**; paired stratified95% interval **{comparison['paired_stratified_95_interval_percentage_points']}pp** (10,000 paired case resamples within category,seed42). This is an unweighted component pilot,not the official BFCLv4 overall score; it is nongating and does not establish population equivalence.",
        "",
        "Only Eight was generated in this campaign. Original scores/outputs are unchanged and reused from the prior audited native campaign. The identical fixed seed42 manifest contains26 single-turn cases (two from each of13 categories) and24 multi-turn cases (six from each of four categories); web-search,memory and format-sensitivity are excluded. Official prompting/parsers/simulators/checkers at commit`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` are unchanged.",
        "",
        "The model is Qwen/Qwen3.8-27B,native RTN W4A16 group128,FP16 activation/KV,FP32 recurrent state. All eight BF16 selectors at zero-based layers3/7/23/27/31/35/39/63 use checkpoint640 weights. Sparse QSA is enabled for prefill and decode; one resident full-model native session and one EP dispatch per forward. No training,calibration or kernel rebuild. Greedy generation uses the actual Qwen chat template,thinking disabled,both official EOS IDs,cap1024 per response,capacity32768. Official step-limit20 semantics remain unchanged (a stopped turn can execute21 calls). No stopped/capped cases are replaced and no output-format repair is applied.",
        "",
        "## Measured workload",
        "",
        "| Metric | Reused Original | Eight |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("generation_calls", "Model calls"),
        ("capped_generations", "Response-token caps"),
        ("context_budget_reductions", "Reduced context budgets"),
        ("step_limited_cases", "Official step-limited cases"),
        ("actual_pruned_cases", "Actual QSA-pruned cases"),
    ]:
        lines.append(
            f"| {label} | {variants['Original'][key]} | {variants['Eight'][key]} |"
        )
    for key, label in [
        ("prompt_tokens", "Prompt tokens/call"),
        ("completion_tokens", "Completion tokens/call"),
        ("declared_turns", "Declared turns/case"),
        ("observed_turns", "Observed turns/case"),
        ("calls_per_case", "Calls/case"),
    ]:
        cells = [
            " / ".join(
                f"{variants[v][key][m]:g}"
                for m in ["minimum", "median", "p95", "maximum"]
            )
            for v in ["Original", "Eight"]
        ]
        lines.append(f"| {label},min / median / p95 / max | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            "Prompt totals include repeatedly processed chat templates,tool schemas,history and tool results. Completions include a generated EOS. A turn is a benchmark user round; each turn can contain many model/tool calls. Actual calls and histories may differ between variants,so elapsed evaluation time is not a fixed-work speed comparison.",
            "",
            "## Frozen initial-prompt groups",
            "",
            "| Group | Cases | Original correct | Eight correct |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, name in [
        ("pruned", "Initial prefill reaches pruning"),
        ("full_support", "Initial prefill stays within full support"),
    ]:
        a = variants["Original"]["initial_groups"][label]
        b = variants["Eight"]["initial_groups"][label]
        assert a["cases"] == b["cases"]
        lines.append(f"| {name} | {a['cases']} | {a['correct']} | {b['correct']} |")
    lines.extend(
        [
            "",
            "The grouping is frozen before generation; later-history pruning coverage is retained separately. First pruning occurs at query position2052 (zero-based). Every new prompt was independently retokenized,every completion independently decoded,and all50 new rows independently rescored with official checkers. Original retains its prior audit; protocol/queue/source hashes bind reused and new evidence separately.",
            "",
            "## Component scores",
            "",
            "| Component | Cases | Original correct | Eight correct |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in comparison["categories"]:
        lines.append(
            f"| `{r['category']}` | {r['cases']} | {r['original_correct']} | {r['candidate_correct']} |"
        )
    lines.extend(
        [
            "",
            "**The source checkpoint still misses its fixed32K PPL gate.** This task pilot does not establish accepted long-context quality. Eight-selector GSM8K and application-memory comparisons remain unmeasured.",
            "",
            "[Paper handoff](../reproducibility/qwen-eight-hip-ep/README.md) · [Summary JSON](../reproducibility/qwen-eight-hip-ep/bfcl-eight-summary.json). The packet's`raw/bfcl-eight-v4-50/` contains per-case,per-generation and per-turn CSVs,official/token audit receipts,and lossless byte-chunked combined row records. The local ZIP also retains complete request messages,rendered prompts,token IDs,all source snapshots,attempt/recovery logs,and referenced Original raw requests. Infrastructure failures are preserved separately and are never scored as wrong answers. Source and portable evidence are versioned on `users/rajeevp/qsa`; the large local ZIP archives remain local.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    files = {}
    for p in TASK.rglob("*"):
        if p.is_file() and p.suffix != ".zip" and "__pycache__" not in p.parts:
            files["project/" + p.relative_to(ROOT).as_posix()] = p
    for row in original:
        queue = ROOT / row["queue"]
        for identifier in row["request_ids"]:
            for suffix in [".request.done", ".response.json"]:
                p = queue / (identifier + suffix)
                files["project/" + p.relative_to(ROOT).as_posix()] = p
        for p in [queue / "attempt.json", *sorted((queue / "source").rglob("*.py"))]:
            files["project/" + p.relative_to(ROOT).as_posix()] = p
    files["source/report_bfcl_eight.py"] = Path(__file__)
    inventory = {
        name: dict(sha256=sha(p), bytes=p.stat().st_size) for name, p in files.items()
    }
    archive = TASK / (
        "evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip"
    )
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in files.items():
            z.write(p, name)
        z.writestr("MANIFEST.json", json.dumps(inventory, indent=2) + "\n")
    with zipfile.ZipFile(archive) as z:
        for name, info in inventory.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"]
    save(
        PACKET / "bfcl-eight-archive.json",
        dict(
            path=archive.relative_to(ROOT).as_posix(),
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            members=len(inventory) + 1,
            published=False,
        ),
    )
    save(
        PACKET / "bfcl-eight-status.json",
        dict(
            state="complete",
            cases=50,
            correct=variants["Eight"]["correct"],
            original_reused=True,
            generation_and_audits_complete=True,
        ),
    )
    print(
        json.dumps(
            dict(
                correct=comparison["correct"],
                delta=comparison["delta_percentage_points"],
                interval=comparison["paired_stratified_95_interval_percentage_points"],
                generations=variants["Eight"]["generation_calls"],
                archive=str(archive),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
