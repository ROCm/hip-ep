#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Render source-backed results for the single-session native QSA integration."""

import hashlib
import json
import math
import re
import subprocess

from prepare_baseline import selector_fingerprint
from prepare_baseline import REFERENCE, ROOT, SELECTOR_SHA, WORK, sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    task = WORK / "native-gqa"
    build = load(task / "build-receipt.json")
    for name, expected in build["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in build["binary_sha256"].items():
        assert sha(task / "install/bin" / name) == expected, name
    validation = load(task / "validation.json")
    assert validation["status"] == "passed"
    assert load(task / "error-validation.json")["status"] == "passed"
    assert selector_fingerprint() == SELECTOR_SHA
    for variant in ["Original", "Changed"]:
        assert (
            validation["dispatch_profiles"][variant]["ep_dispatches_per_forward"] == 1
        )
        folder = task / "models" / variant
        export = load(folder / "export.json")
        for name, info in export["artifacts"].items():
            assert sha(folder / name) == info["sha256"]
    candidates = []
    for context in [4096, 16384, 32768, 65472]:
        folder = task / "paired-performance" / str(context)
        if not (folder / "status.json").exists():
            continue
        status, summary, protocol = [
            load(folder / (n + ".json")) for n in ["status", "summary", "protocol"]
        ]
        assert (
            status["status"] == "complete"
            and protocol["ep_dispatches_per_forward"] == 1
        )
        rows = [
            json.loads(line)
            for line in (folder / "rows.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2 * summary["pairs"]
        for variant in ["Original", "Changed"]:
            selected = [r for r in rows if r["variant"] == variant]
            assert {r["pair"] for r in selected} == set(range(summary["pairs"]))
            assert sum(r["order"] == 0 for r in selected) == summary["pairs"] // 2
            for phase in ["prefill", "decode"]:
                actual = sum(r[phase + "_seconds"] for r in selected) / len(selected)
                assert (
                    abs(actual - summary["phases"][phase][variant]["mean_seconds"])
                    < 1e-9
                )
        candidates.append(summary)
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# OLMo QSA: native GQA integration",
        "",
        "Both variants run one complete ONNX graph in one session. The native QSA specialization replaces only attention middles 7/11/15/23. Layer 19 stays dense. All 224 backbone projection nodes and packed quantized weights remain unchanged. Batch is one; selectors are unchanged BF16; training updates are zero.",
        "",
        "## Execution and correctness",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Dense Original full graph versus dense boundary reference | Passed |",
        "| Native QSA full graph versus frozen boundary reference | Passed |",
        "| Prefixes 17/517/1025/4092 plus four cached calls | Finite; exact native QSA replay |",
        "| Native QSA full 4K versus cached 4092+4 | Passed |",
        "| ORT dispatches per forward | Original: 1; Changed: 1 |",
        "| Frozen selector and persistent cache checks | Passed |",
        "",
        "GPU validation uses the same new EP build for both complete models and their reference backbone. The selector math remains BF16 input/parameters/output with FP32 accumulation. Decode uses a native GEMV; prefill uses native BF16 hipBLASLt. There are no Torch/ctypes/session handoffs at attention boundaries. Internal staged HIP kernels still perform the additional selector and sparse-attention work.",
        "",
    ]
    quality_path = task / "paired-quality/dolma-summary.json"
    assert quality_path.exists(), "Complete the retained paired native Dolma gate first"
    assert any(s["context"] == 4096 for s in candidates), (
        "Complete native 4K timing first"
    )
    if quality_path.exists():
        quality = load(quality_path)
        assert quality["complete"]
        raw_quality = task / "paired-quality/dolma.jsonl"
        rows = [json.loads(line) for line in raw_quality.read_text().splitlines()]
        assert len(rows) == 128
        assert {(r["variant"], r["document"]) for r in rows} == {
            (v, i) for v in ["Original", "Changed"] for i in range(64)
        }
        for variant in ["Original", "Changed"]:
            selected = [r for r in rows if r["variant"] == variant]
            assert all(
                r["targets"] == 4095 and r["cached_targets"] == 64 for r in selected
            )
            ppl = math.exp(sum(r["nll_sum"] for r in selected) / (64 * 4095))
            assert abs(ppl - quality["variants"][variant]["ppl"]) < 1e-9
        a, b = [quality["variants"][v]["ppl"] for v in ["Original", "Changed"]]
        assert abs(100 * (b / a - 1) - quality["ppl_change_percent"]) < 1e-9
        ci = quality["paired_95_interval_percent"]
        gate = "Pass" if quality["gate_point_estimate_pass"] else "Fail"
        lines += [
            "## Paired Dolma quality",
            "",
            "| Original PPL | Changed PPL | Change | Paired 95% interval | 5% gate |",
            "|---:|---:|---:|---|---|",
            f"| {a:.6f} | {b:.6f} | {quality['ppl_change_percent']:+.3f}% | [{ci[0]:+.3f}, {ci[1]:+.3f}]% | {gate} |",
            "",
            "Same 64 held-out 4096-token windows, 4095 prefill targets and 64 cached targets per document. These windows are reused from the source project. Original/Changed order alternates by document. The gate is relative to native Original in this build. This does not establish broad long-context quality.",
            "",
        ]
    else:
        lines += ["Paired native Dolma evaluation is pending.", ""]
    gsm_rows = []
    gsm_path = task / "gsm8k/gsm8k-summary.json"
    if gsm_path.exists() and load(gsm_path)["complete"]:
        gsm = load(gsm_path)
        assert gsm["acceptance_gate_enabled"] is False
        gsm_rows = [
            json.loads(line)
            for line in (gsm_path.parent / "gsm8k.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(gsm_rows) == 256
        assert {(r["variant"], r["index"]) for r in gsm_rows} == {
            (v, i) for v in ["Original", "Changed"] for i in range(128)
        }
        source = REFERENCE / "data/gsm8k.json"
        assert (
            sha(source) == load(REFERENCE / "data/inputs.json")["sha256"]["gsm8k.json"]
        )
        data = load(source)
        prefix = "\n\n".join(
            f"Question: {r['question']}\nAnswer: {r['answer']}" for r in data["fewshot"]
        )

        def answer(text):
            match = re.search(r"####\s*(-?[0-9.,]+)", text)
            return match.group(1).replace(",", "").rstrip(".") if match else None

        for row in gsm_rows:
            item = data["items"][row["index"]]
            assert (
                row["prompt"]
                == prefix + "\n\nQuestion: " + item["question"] + "\nAnswer:"
            )
            assert row["gold"] == answer(item["answer"]) and row["gold"] is not None
            assert row["prediction"] == answer(
                row["completion"].split("Question:", 1)[0]
            )
            assert row["correct"] == (row["prediction"] == row["gold"])
        for variant in ["Original", "Changed"]:
            selected = [r for r in gsm_rows if r["variant"] == variant]
            assert (
                sum(r["correct"] for r in selected)
                == gsm["variants"][variant]["correct"]
            )
            assert len(selected) == gsm["variants"][variant]["items"] == 128
        a, b = [gsm["variants"][v] for v in ["Original", "Changed"]]
        ci = gsm["paired_95_interval_percentage_points"]
        lines += [
            "## Native GSM8K pilot",
            "",
            "| Default | QSA | Change | Paired 95% interval | Acceptance gate |",
            "|---:|---:|---:|---|---|",
            f"| {a['correct']}/128 ({a['accuracy_percent']:.2f}%) | {b['correct']}/128 ({b['accuracy_percent']:.2f}%) | {gsm['accuracy_change_percentage_points']:+.3f} pp | [{ci[0]:+.3f}, {ci[1]:+.3f}] pp | Disabled |",
            "",
            "Fresh native generations on the retained 128-question five-shot pilot, alternating variant order. Both variants use greedy generation, EOS/Question stopping, a 512-token cap and the same strict #### parser. All 256 prompts were verified against the UTF-8 source, and answers/scores were recomputed from raw completions. Three generations per variant reached the cap. This is a pilot, not the full GSM8K test split; it exactly reproduces the previously accepted 94/128 versus 91/128 scores.",
            "",
        ]
    for phase in ["decode", "prefill"]:
        lines += [
            f"## {phase.capitalize()}, separate timer",
            "",
            "| Prefix tokens | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |",
            "|---:|---:|---:|---:|---|---:|",
        ]
        for summary in candidates:
            p = summary["phases"][phase]
            ci = p["paired_95_interval_percent"]
            lines.append(
                f"| {summary['context']:,} | {p['Original']['tokens_s']:.4f} | {p['Changed']['tokens_s']:.4f} | {p['throughput_change_percent']:+.3f}% | [{ci[0]:+.3f}, {ci[1]:+.3f}]% | {summary['pairs']} |"
            )
        lines.append("")
    lines += [
        "Rates use balanced repeated Original/Changed pairs on the same physical GPU with identical token inputs, packed backbone weights, head work and persistent KV buffers. Two exact-shape warmups per variant verify finite outputs and exact replay. Timers are HIP-synchronized CPU wall time. Decode covers 64 fixed teacher-forced calls and excludes prefill, sampling, model loading and tensor/binding allocation. Prefill includes the final-position output head and excludes model loading and preallocation. Intervals use 10,000 paired bootstrap resamples with seed 42 and describe within-run variation. Aggregate and per-request rates are equal at batch one.",
        "",
        "GSM8K remains diagnostic and is not an acceptance gate, as requested. Earlier multi-session timings and the original pilot remain in the [boundary prototype report](olmo-qsa-results.md). Native accuracy and timing results are included here only after their own runs complete.",
        "",
        "The ONNX frontend originally lacked BF16 constant type mapping. The native integration adds that mapping and a raw-byte dtype regression check, preserving selector storage instead of converting parameters. Compiler tests also cover native lowering and unsupported dtype/geometry rejection.",
        "",
        "## Evidence",
        "",
        "Hardware is the same Radeon 8060S gfx1151 / Ryzen AI MAX+ PRO 395 Windows machine used for the boundary prototype. The native runtime is built from this branch into a separate installation. Build commands, compiler/runtime versions, immutable export receipts, binary/source hashes, profiles and raw rows are retained under `_local/qsa/native-gqa/` and `_local/qsa/stages/`.",
        "",
        "Validation passed 608 compiler LIT cases (9 unsupported), three runtime tests, three constant/dtype CPU tests and five Python export-contract tests. The final audit verifies the exact built source and installed binary hashes from `build-receipt.json`, as well as all 128 native Dolma rows and every completed timing pair.",
        "",
        "See the [native operator design](../design/olmo-qsa-native-gqa.md) for its fixed geometry, cache/state contract and reproduction entry points. No support is claimed beyond the validated batch-one geometry or measured contexts.",
        "",
    ]
    output = ROOT / "docs/experiments/olmo-qsa-native-gqa-results.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    receipt = {
        "status": "passed",
        "batch": 1,
        "training_updates": 0,
        "checkout_at_report": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "build_checkout_base": build["build_checkout_base"],
        "build_receipt_sha256": sha(task / "build-receipt.json"),
        "selector_sha256": SELECTOR_SHA,
        "source_sha256": build["source_sha256"],
        "binary_sha256": build["binary_sha256"],
        "measured_contexts": [r["context"] for r in candidates],
        "dolma_gate_pass": quality["gate_point_estimate_pass"],
        "gsm8k_rows_verified": len(gsm_rows),
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (task / "final-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
